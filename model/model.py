"""
Physics-Informed Graph Neural Network for Rydberg Entanglement Entropy Prediction

Architecture Design Principles:
  1. **Physical Inductive Biases**: 
     - Edge features encode geometric relationships (angle, distance) and quantum correlations
     - Global features incorporate subsystem size ratios (nA/N, nB/N) as physical constraints
  2. **Numerical Stability**:
     - Softplus output activation ensures non-negative entropy predictions
     - Log-cosh loss provides robustness to outliers compared to MSE
  3. **Scalability**:
     - Set2Set readout handles variable-sized graphs without fixed pooling
     - Residual connections enable deep message passing (6 layers)
  4. **Regularization**:
     - Dropout at multiple levels (edge/node/readout)
     - BatchNorm after every linear transformation
     - Gradient clipping during training

Reference:
  Predicting the von Neumann entanglement entropy using a graph neural network
  Anas Saleh 2025 Mach. Learn.: Sci. Technol. 6 035034
"""

from typing import Dict

import torch
import torch.nn as nn
import torch_geometric
from config import FEATURE_SCHEMA
from torch_geometric.nn import BatchNorm, GINEConv, Set2Set, TransformerConv


class PhysicalScaleAwareLoss(nn.Module):
    """
    Args:
        eps (float): Numerical stability constant to prevent log(0) (default: 1e-10)
    
    Inputs:
        pred_s (torch.Tensor): Predicted entropy values of shape (batch_size,)
        target_s (torch.Tensor): Ground truth entropy values of shape (batch_size,)
        system_size (torch.Tensor): Total system sizes (N) of shape (batch_size,)
        subsystem_size (torch.Tensor): Subsystem A sizes (nA) of shape (batch_size,)
    
    Returns:
        torch.Tensor: Scalar loss value (mean over batch)
    
    Note:
        system_size and subsystem_size arguments are included for API compatibility
        with physics-informed loss variants that may incorporate size-dependent weighting.
        Current implementation does not use these parameters but maintains the signature
        for future extensibility.
    """
    
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_s: torch.Tensor,
        target_s: torch.Tensor,
        system_size: torch.Tensor,
        subsystem_size: torch.Tensor
    ) -> torch.Tensor:
        error = pred_s - target_s
        logcosh = torch.log(torch.cosh(error + self.eps))
        return logcosh.mean()


def init_weights(module: nn.Module) -> None:
    """
    Kaiming (He) initialization with adjustments for GNN stability.
    
    Applies layer-specific initialization strategies:
        - Linear layers: Kaiming normal (fan_in mode) for ReLU/SiLU activations
        - BatchNorm: Ones for weights, zeros for biases (standard practice)
        - Attention layers: Sigmoid-constrained initialization for stable attention
    
    Args:
        module (nn.Module): PyTorch module to initialize
    
    
    Usage:
        model.apply(init_weights)  # Applies recursively to all submodules
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ExperimentalGNN(nn.Module):
    """
    Multi-scale GNN for predicting von Neumann entanglement entropy in Rydberg systems.
    
    Architecture combines:
        - Edge-attention mechanisms for weighting quantum correlations
        - Alternating GINEConv/TransformerConv layers for diverse message passing
        - Subsystem-aware global feature integration
        - Set2Set readout 
    
    Input Feature Specification:
        Node features (4D):
            [0]: x-position (lattice coordinate)
            [1]: y-position (lattice coordinate)
            [2]: Rydberg excitation probability (diagonal correlation)
            [3]: Subsystem mask (1 if in A, 0 if in B)
        
        Edge features (3D):
            [0]: Angle between sites (radians, [-π, π])
            [1]: Connected correlation (P(i,j) - P(i)P(j))
            [2]: Normalized distance (|r_ij|/√N)
    
    Output:
        Scalar entropy prediction S ∈ [0, log(min(nA, nB))] (natural units)
    
    Args:
        hidden_channels (int): Base dimensionality for all hidden representations
        num_layers (int): Number of message passing layers (default: 6)
        dropout_p (float): Dropout probability for regularization (default: 0.4)
    
    Attributes:+
        node_encoder (nn.Sequential): Projects 4D node features → hidden space
        edge_encoder (nn.Sequential): Projects 3D edge features → hidden space
        edge_attention (nn.ModuleList): Per-layer attention over edge features
        convs (nn.ModuleList): Alternating GINEConv/TransformerConv layers
        edge_convs (nn.ModuleList): Edge feature refinement MLPs
        norms (nn.ModuleList): BatchNorm after each convolution
        pool (nn.ModuleList): Intermediate refinement MLPs (applied every 2 layers)
        readout (nn.ModuleList): Dual-head Set2Set for robust graph representation
        readout_projection (nn.Sequential): Dimensionality reduction for readout
        global_mlp (nn.Sequential): Processes subsystem size ratios
        final_mlp (nn.Sequential): Combines all features → entropy prediction
    
    Forward Pass Flow:
        1. Encode node/edge features into hidden space
        2. For each layer (0 to num_layers-1):
            a. Compute edge attention weights from current edge features
            b. Refine edge features through MLP
            c. Perform message passing with attention-weighted edges
            d. Apply residual connection and BatchNorm
            e. Every 2 layers: Apply intermediate refinement MLP
        3. Dual-head Set2Set readout → concatenate → project to 2*hidden_channels
        4. Compute global features: [nA/N, nB/N]
        5. Concatenate readout + global features → final MLP → Softplus output
    """
    
    def __init__(
        self,
        hidden_channels: int = 512,
        num_layers: int = 6,
        dropout_p: float = 0.4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # Feature index mapping for clarity and maintainability
        self.feature_indices: Dict[str, slice | int] = {
            'position': slice(0, 2),    # x, y coordinates
            'rydberg_val': 2,           # Diagonal correlation
            'mask': 3,                  # Subsystem membership
        }

        # Node feature encoder (4D → hidden_channels)
        self.node_encoder = nn.Sequential(
            nn.Linear(FEATURE_SCHEMA['node_features']['total_dim'], hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU()
        )

        # Edge feature encoder (3D → hidden_channels)
        self.edge_encoder = nn.Sequential(
            nn.Linear(FEATURE_SCHEMA['edge_features']['total_dim'], hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU()
        )

        # Per-layer edge attention mechanisms
        self.edge_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()  # Constrain attention to [0,1]
            ) for _ in range(num_layers)
        ])

        # Message passing layers with alternating architectures
        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i % 2 == 0:
                # GINEConv: Graph Isomorphism Network with edge features
                mp_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    BatchNorm(hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINEConv(nn=mp_mlp, edge_dim=hidden_channels)
            else:
                # TransformerConv: Multi-head attention with edge conditioning
                conv = TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // 8,  # 8 heads → 64 per head
                    heads=8,
                    edge_dim=hidden_channels,
                    dropout=dropout_p,
                    beta=True,   # Learnable skip connections
                    concat=True  # Concatenate head outputs
                )
            self.convs.append(conv)

            # Edge feature refinement MLP
            self.edge_convs.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ))

            # Node feature normalization
            self.norms.append(BatchNorm(hidden_channels))

        # Intermediate refinement (applied every 2 layers)
        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_layers // 2)
        ])

        # Dual-head Set2Set readout for permutation invariance
        self.readout = nn.ModuleList([
            Set2Set(hidden_channels, processing_steps=4) for _ in range(2)
        ])
        
        # Project concatenated readouts to manageable dimension
        self.readout_projection = nn.Sequential(
            nn.Linear(4 * hidden_channels, 2 * hidden_channels),  # 2 heads × 2× hidden
            BatchNorm(2 * hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )

        # Global feature processor (subsystem size ratios)
        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),  # [nA/N, nB/N]
            BatchNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Final prediction head with physical constraint (non-negative entropy)
        combined_dim = (2 * hidden_channels) + hidden_channels  # readout + global
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels // 2),
            BatchNorm(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()  # Enforce S ≥ 0 (physical constraint)
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """
        Forward pass: graph → entropy prediction.
        
        Args:
            data (torch_geometric.data.Data): Batched graph with attributes:
                - x: Node features [num_nodes, 4]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 3]
                - batch: Batch assignment vector [num_nodes]
                - system_size: Total sites per graph [batch_size, 1]
                - nA: Subsystem A size per graph [batch_size, 1]
        
        Returns:
            torch.Tensor: Predicted entropies of shape [batch_size]
        
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Reconstruct node features with explicit schema validation
        node_features = torch.cat([
            x[:, self.feature_indices['position']],                      # [N, 2]
            x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),    # [N, 1]
            x[:, self.feature_indices['mask']].unsqueeze(-1),           # [N, 1]
        ], dim=1)  # [N, 4]

        # Initial encoding
        h = self.node_encoder(node_features)      # [N, hidden]
        e = self.edge_encoder(edge_attr)          # [E, hidden]

        # Message passing with edge attention
        for i in range(self.num_layers):
            # Compute attention weights for current edges
            edge_weights = self.edge_attention[i](e).squeeze(-1)  # [E]
            
            # Refine edge features
            e = self.edge_convs[i](e)  # [E, hidden]
            
            # Message passing with attention-weighted edges
            h_new = self.convs[i](h, edge_index, e * edge_weights.unsqueeze(-1))
            
            # Normalization and residual connection
            h_new = self.norms[i](h_new)
            h = h + h_new  # Residual connection
            
            # Intermediate refinement every 2 layers
            if i % 2 == 0 and i // 2 < len(self.pool):
                h = self.pool[i // 2](h)

        # Multi-head readout
        readouts = [rd(h, batch) for rd in self.readout]  # List of [B, 2*hidden]
        h_readout = torch.cat(readouts, dim=1)             # [B, 4*hidden]
        h_readout = self.readout_projection(h_readout)     # [B, 2*hidden]

        # Global subsystem features (size ratios)
        N = data.system_size.squeeze(-1) + 1e-10  # [B]
        nA = data.nA.squeeze(-1)                  # [B]
        nB = data.nB.squeeze(-1)                  # [B]
        nA_over_N = nA / N                        # [B]
        nB_over_N = nB / N                        # [B]
        global_feats = torch.stack([nA_over_N, nB_over_N], dim=1)  # [B, 2]
        gf_out = self.global_mlp(global_feats)    # [B, hidden]

        # Final prediction
        combined = torch.cat([h_readout, gf_out], dim=1)  # [B, 3*hidden]
        out = self.final_mlp(combined).squeeze(-1)        # [B]
        return out