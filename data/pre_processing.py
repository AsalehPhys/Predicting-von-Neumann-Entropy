"""
Rydberg Atom Graph Dataset Processing Pipeline

Converts raw Rydberg atom bitstring data (stored in Parquet format) into
PyTorch Geometric graph representations suitable for graph neural network training.

The pipeline:
1. Loads simulation data containing quantum states and subsystem information
2. Computes quantum correlations 
3. Constructs graph representations 
4. Processes data in parallel chunks with fault tolerance and intermediate checkpointing
5. Saves final dataset in PyTorch Geometric's InMemoryDataset format


"""

import gc
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, InMemoryDataset

from data.config_preprocessing import CONFIG, set_dynamic_config


def setup_logging() -> None:
    """
    Configures the root logger with standardized formatting and output destination.
    
    Sets up logging to output messages to stdout with the following format:
        YYYY-MM-DD HH:MM:SS [LEVEL] message
    
    Log levels used in this module:
        - INFO: Normal processing progress and statistics
        - WARNING: Recoverable errors (e.g., single row processing failures)
        - ERROR: Critical failures requiring intervention
    
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def set_seed(seed: int) -> None:
    """
    Sets random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed (int): Integer seed value to initialize random number generators.
    

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and CONFIG['use_gpu']:
        torch.cuda.manual_seed_all(seed)


def calculate_quantum_correlations_optimized(
    state_indices: Union[List[int], np.ndarray],
    state_probs: Union[List[float], np.ndarray],
    N: int
) -> np.ndarray:
    """
    Computes diagonal Rydberg excitation probabilities for each site in the lattice.
    
    For each site i, calculates P_i = Σ_{states where site i is excited} P(state)
    This represents the marginal probability of finding a Rydberg excitation at site i.
    
    Args:
        state_indices (Union[List[int], np.ndarray]): 
            List or array of integer state indices where each bit represents 
            occupation (1) or vacancy (0) of a lattice site. Length = number of states.
        state_probs (Union[List[float], np.ndarray]): 
            List or array of probabilities corresponding to each state index. 
            Must sum to 1. Length must match state_indices.
        N (int): Total number of lattice sites in the system (Ny * Nx).
    
    Returns:
        np.ndarray: Array of shape (N,) containing diagonal correlation values 
                    (Rydberg excitation probabilities) for each site.
    

    """
    diag_correlations = np.zeros(N, dtype=np.float32)
    states = np.array(state_indices, dtype=np.int64)
    probs = np.array(state_probs, dtype=np.float32)
    
    for i in range(N):
        mask = (states & (1 << i)) != 0
        diag_correlations[i] = np.sum(probs[mask])
    
    return diag_correlations


def create_edges(positions: np.ndarray, N: int) -> np.ndarray:
    """
    Generates a fully connected edge index for the lattice graph.
    
    Creates edges between all pairs of distinct sites (complete graph topology).
    
    Args:
        positions (np.ndarray): Array of shape (N, 2) containing 2D coordinates of each lattice site.
        N (int): Total number of lattice sites in the system.
    
    Returns:
        np.ndarray: Edge index array of shape (2, E) where E = N*(N-1)/2, 
                    with edge_index[0] containing source nodes and 
                    edge_index[1] containing target nodes.
    
    Implementation Notes:
        - Edge ordering follows lexicographic ordering of node pairs (i, j) where i < j
    """
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append([i, j])
    return np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)


def process_single_row(row_data: pd.Series) -> Optional[Dict[str, np.ndarray]]:
    """
    Converts a single simulation row into a graph representation with node/edge features.
    
    Processes quantum state information to compute:
        - Node features: positions, Rydberg probabilities, subsystem membership
        - Edge features: geometric angles, normalized distances, connected correlations
        - Graph-level targets: von Neumann entropy
    
    Args:
        row_data (pd.Series): Pandas Series containing simulation data with required fields:
            - 'Ny': Number of sites in y-dimension (x-dimension fixed at 2)
            - 'All_Indices': List of state indices (bit representations)
            - 'All_Probabilities': List of state probabilities
            - 'Subsystem_Mask': String of '0's and '1's indicating subsystem membership
            - 'Von_Neumann_Entropy': Target entanglement entropy value
    
    Returns:
        Optional[Dict[str, np.ndarray]]: Dictionary containing graph components as NumPy arrays:
            - 'node_features': (N, 4) array [x_pos, y_pos, rydberg_prob, subsystem_mask]
            - 'edge_index': (2, E) array of edge connections
            - 'edge_attr': (E, 3) array [angle, connected_corr, normalized_dist]
            - 'target_vne': (1,) array containing von Neumann entropy
            - 'system_size': (1, 1) array with total site count N
            - 'nA', 'nB': (1, 1) arrays with subsystem sizes
    
    Raises:
        Exception: Any processing error is caught and logged; function returns None on failure.
    
    """
    try:
        Ny = row_data['Ny']
        Nx = 2
        N = Ny * Nx

        # Generate 2D positions for rectangular lattice (2 rows × Ny columns)
        positions = np.array([(col * 1, row * 2)  
                            for row in range(Nx) 
                            for col in range(Ny)], dtype=np.float32)

        # Compute diagonal Rydberg excitation probabilities per site
        rydberg_probs = calculate_quantum_correlations_optimized(
            row_data['All_Indices'], row_data['All_Probabilities'], N)

        # Parse subsystem mask from string to binary array
        mask = np.array([int(bit) for bit in row_data['Subsystem_Mask']], 
                       dtype=np.float32).reshape(-1, 1)

        # Assemble node features: [x, y, rydberg_prob, subsystem_mask]
        node_features = np.concatenate([
            positions,
            rydberg_probs.reshape(-1, 1),
            mask
        ], axis=1)

        # Generate complete graph edge index
        edge_index = create_edges(positions, N)

        # Compute edge attributes if edges exist
        if edge_index.size > 0:
            pos_i = positions[edge_index[0]]
            pos_j = positions[edge_index[1]]
            vec_ij = pos_j - pos_i
            dist_ij = np.linalg.norm(vec_ij, axis=1, keepdims=True) / np.sqrt(N)
            angle_ij = np.arctan2(vec_ij[:, 1], vec_ij[:, 0]).reshape(-1, 1)
            
            # Compute connected correlations for each edge
            edge_corr_values = []
            states = np.array(row_data['All_Indices'], dtype=np.int64)
            probs = np.array(row_data['All_Probabilities'], dtype=np.float32)
            
            for i, j in zip(edge_index[0], edge_index[1]):
                mask_i = (states & (1 << i)) != 0
                mask_j = (states & (1 << j)) != 0
                mask_ij = mask_i & mask_j
                corr_ij = np.sum(probs[mask_ij])
                connected_corr = corr_ij - rydberg_probs[i] * rydberg_probs[j]
                edge_corr_values.append(connected_corr)
            
            edge_corr_values = np.array(edge_corr_values, dtype=np.float32).reshape(-1, 1)
            edge_attr = np.concatenate([angle_ij, edge_corr_values, dist_ij], axis=1)
        else:
            edge_attr = np.zeros((0, 3), dtype=np.float32)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_vne': np.array([row_data['Von_Neumann_Entropy']], dtype=np.float32),
            'system_size': np.array([[N]], dtype=np.float32),
            'nA': np.array([[float(mask.sum())]], dtype=np.float32),
            'nB': np.array([[float(N - mask.sum())]], dtype=np.float32)
        }
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return None


def process_chunk(chunk_data: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
    """
    Processes a chunk of simulation rows in sequence, converting each to graph representation.
    
    Designed to be executed in a separate process via ProcessPoolExecutor for parallelism.
    
    Args:
        chunk_data (pd.DataFrame): DataFrame containing simulation rows to process.
                                   Must contain all fields required by process_single_row().
    
    Returns:
        List[Dict[str, np.ndarray]]: List of successfully processed graph representations.
    
    """
    results = []
    for _, row in chunk_data.iterrows():
        try:
            data = process_single_row(row)
            if data is not None:
                results.append(data)
        except Exception as e:
            logging.warning(f"Failed to process row: {str(e)}")
    return results


class RydbergDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for Rydberg atom simulation data.
    
    Converts raw quantum simulation data into graph representations suitable for
    graph neural network training.
    
    Attributes:
        df (pd.DataFrame): Filtered and shuffled input DataFrame after size-based sampling.
        processed_dir (str): Directory where processed dataset files are stored.
        processed_file_name (str): Filename for the saved PyTorch dataset.
    
    Inheritance:
        torch_geometric.data.InMemoryDataset: Provides automatic caching and loading
        of processed data via the processed_paths mechanism.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root: str = '.',
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None
    ) -> None:
        """
        Initializes the RydbergDataset with input data and processing configuration.
        
        Args:
            dataframe (pd.DataFrame): Input simulation data containing quantum states
                                      and subsystem information.
            root (str): Root directory where dataset files will be stored. 
                        Processed files saved in root/processed/.
            transform (Optional[Any]): Transformation to apply to each Data object 
                                       after loading (default: None).
            pre_transform (Optional[Any]): Transformation to apply before saving 
                                           to disk (default: None).
        
        Processing Flow:
            1. Applies size-based filtering via CONFIG['max_points_per_ny']
            2. Shuffles filtered data for randomization
            3. Checks for existing processed file at processed_paths[0]
            4. If exists: loads cached data; else: triggers process() method
            5. Stores processed data and slices in self.data and self.slices
        """
        self.df = dataframe
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.process()

    @property
    def raw_file_names(self) -> List[str]:
        """
        Required override for InMemoryDataset. No raw files used in this implementation.
        
        Returns:
            List[str]: Empty list since data is provided directly via DataFrame.
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        Specifies the filename for the processed dataset cache.
        
        Returns:
            List[str]: Single-element list containing CONFIG['processed_file_name'].
        """
        return [CONFIG['processed_file_name']]

    def download(self) -> None:
        """
        Required override for InMemoryDataset. No download functionality needed.
        
        This dataset uses locally provided Parquet files rather than remote downloads.
        """
        pass

    def numpy_to_torch_data(
        self,
        numpy_data: Dict[str, np.ndarray],
        device: str = 'cpu'
    ) -> Optional[Data]:
        """
        Converts NumPy graph representation to PyTorch Geometric Data object.
        
        Args:
            numpy_data (Dict[str, np.ndarray]): Dictionary containing graph components:
                - 'node_features': (N, F) array of node features
                - 'edge_index': (2, E) array of edge connections
                - 'edge_attr': (E, D) array of edge features
                - 'target_vne': (1,) array of target values
                - 'system_size', 'nA', 'nB': Graph-level attributes
            device (str): Device to place tensors on ('cpu' or 'cuda') (default: 'cpu').
        
        Returns:
            Optional[Data]: PyTorch Geometric Data object with attributes:
                - x: Node feature matrix (N, F)
                - edge_index: Edge connectivity (2, E)
                - edge_attr: Edge feature matrix (E, D)
                - y: Target values (1,)
                - system_size, nA, nB: Additional graph-level attributes
        
        Returns None if conversion fails due to incompatible array shapes or types.
        
        Implementation Notes:
            - All arrays converted to appropriate dtypes (float32 for features, int64 for indices)
            - Tensors moved to specified device only if device != 'cpu' (avoids unnecessary transfers)
            - Preserves all graph-level attributes as separate tensor properties
        """
        try:
            data = Data(
                x=torch.from_numpy(numpy_data['node_features']).float(),
                edge_index=torch.from_numpy(numpy_data['edge_index']).long(),
                edge_attr=torch.from_numpy(numpy_data['edge_attr']).float(),
                y=torch.from_numpy(numpy_data['target_vne']).float(),
                system_size=torch.from_numpy(numpy_data['system_size']).float(),
                nA=torch.from_numpy(numpy_data['nA']).float(),
                nB=torch.from_numpy(numpy_data['nB']).float()
            )
            return data.to(device) if device != 'cpu' else data
        except Exception as e:
            logging.error(f"Error converting numpy to torch: {str(e)}")
            return None

    def process(self) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """
        Main processing pipeline: converts raw DataFrame into PyTorch Geometric format.
        
        Processing stages:
            1. Size-based stratified sampling (per Ny value)
            2. Global shuffling of filtered dataset
            3. Chunking into manageable processing units
            4. Parallel batch processing with timeout protection
            5. Intermediate checkpointing every save_interval chunks
            6. Conversion to PyTorch Geometric Data objects
            7. Collation into single InMemoryDataset format
            8. Cleanup of temporary checkpoint files
        
        Returns:
            Tuple[Data, Dict[str, torch.Tensor]]: 
                - data: Concatenated Data object containing all graphs
                - slices: Dictionary mapping attribute names to slice indices
        
        Raises:
            RuntimeError: If no data is successfully processed after all attempts.
            Exception: Any unhandled exception during processing (logged before re-raising).
        
        Fault Tolerance Features:
            - Per-chunk timeouts (CONFIG['timeout'] seconds)
            - Individual row failure isolation (failed rows skipped)
            - Intermediate checkpointing to prevent total data loss on crash
            - Memory cleanup via gc.collect() between batches
            - Temporary file cleanup on successful completion
        
        """
        try:
            # Apply size-based stratified sampling
            filtered_data = []
            for ny, max_points in CONFIG['max_points_per_ny'].items():
                ny_data = self.df[self.df['Ny'] == ny]
                if len(ny_data) > max_points:
                    ny_data = ny_data.sample(max_points, random_state=CONFIG['random_seed'])
                filtered_data.append(ny_data)
            
            self.df = pd.concat(filtered_data, ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

            # Split into chunks for parallel processing
            num_chunks = len(self.df) // CONFIG['chunk_size'] + 1
            chunks = np.array_split(self.df, num_chunks)
            
            processed_chunks: List[Dict[str, np.ndarray]] = []
            total_processed = 0
            start_time = time.time()
            last_save = start_time
            
            logging.info(f"Starting processing with {len(chunks)} chunks "
                        f"(chunk size: {CONFIG['chunk_size']}, workers: {CONFIG['num_workers']})")
            
            # Process chunks in batches to reduce process spawn overhead
            for batch_idx in range(0, len(chunks), CONFIG['batch_size']):
                batch_start = time.time()
                batch_chunks = chunks[batch_idx:batch_idx + CONFIG['batch_size']]
                current_batch_num = batch_idx // CONFIG['batch_size'] + 1
                total_batches = (len(chunks) - 1) // CONFIG['batch_size'] + 1
                
                logging.info(f"\nStarting batch {current_batch_num}/{total_batches} "
                           f"(chunks {batch_idx+1} to {min(batch_idx+CONFIG['batch_size'], len(chunks))})")
                logging.info(f"Total rows processed so far: {total_processed}")
                
                # Process batch in parallel with timeout protection
                with ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
                    future_to_chunk = {
                        executor.submit(process_chunk, chunk): i 
                        for i, chunk in enumerate(batch_chunks)
                    }
                    
                    for future in as_completed(future_to_chunk):
                        chunk_idx = batch_idx + future_to_chunk[future]
                        try:
                            chunk_results = future.result(timeout=CONFIG['timeout'])
                            if chunk_results:
                                processed_chunks.extend(chunk_results)
                                total_processed += len(chunk_results)
                                
                                logging.info(
                                    f"Chunk {chunk_idx + 1}/{len(chunks)} complete. "
                                    f"Processed {len(chunk_results)} rows "
                                    f"(Total: {total_processed})"
                                )
                                
                                current_time = time.time()
                                if (current_time - last_save > 600 or 
                                    (chunk_idx + 1) % CONFIG['save_interval'] == 0):
                                    temp_save_path = os.path.join(
                                        self.processed_dir, 
                                        f'temp_numpy_chunk_{chunk_idx+1}.npy'
                                    )
                                    np.save(temp_save_path, processed_chunks)
                                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                                    logging.info(
                                        f"Saved intermediate results at chunk {chunk_idx+1}. "
                                        f"Memory usage: {memory_usage:.2f} MB"
                                    )
                                    last_save = current_time
                            else:
                                logging.warning(f"Chunk {chunk_idx + 1} returned no results")
                                
                        except FuturesTimeoutError:
                            logging.error(f"Timeout processing chunk {chunk_idx + 1} "
                                        f"(exceeded {CONFIG['timeout']}s)")
                            continue
                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                            continue
                
                batch_time = time.time() - batch_start
                avg_time = batch_time / len(batch_chunks) if batch_chunks else 0
                logging.info(
                    f"Batch {current_batch_num} complete. Time: {batch_time:.2f}s. "
                    f"Average per chunk: {avg_time:.2f}s"
                )
                
                # Force garbage collection to manage memory in long-running processes
                gc.collect()
            
            logging.info("Processing complete. Converting to PyTorch Geometric format...")
            
            if not processed_chunks:
                raise RuntimeError("No data was successfully processed after all attempts")
            
            # Convert NumPy representations to PyTorch Geometric Data objects
            data_list: List[Data] = []
            for numpy_data in processed_chunks:
                torch_data = self.numpy_to_torch_data(numpy_data)
                if torch_data is not None:
                    data_list.append(torch_data)
            
            if not data_list:
                raise RuntimeError("No Data objects created during conversion step")
            
            # Clean up temporary checkpoint files
            for temp_file in os.listdir(self.processed_dir):
                if temp_file.startswith('temp_numpy_chunk_'):
                    try:
                        os.remove(os.path.join(self.processed_dir, temp_file))
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            
            # Collate into InMemoryDataset format and save
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.2f}s "
                       f"({total_time/total_processed:.4f}s per sample)")
            logging.info(f"Processed {total_processed} rows successfully "
                       f"({len(data_list)} valid graphs)")
            
            return data, slices
            
        except Exception as e:
            logging.error(f"Error in process method: {str(e)}")
            raise


def load_data() -> RydbergDataset:
    """
    Loads and preprocesses Rydberg simulation data from Parquet files.
    
    Workflow:
        1. Validates existence of all files in CONFIG['data_paths']
        2. Reads Parquet files into Pandas DataFrames using PyArrow
        3. Concatenates multiple files into single DataFrame
        4. Applies size-based sampling per Ny value
        5. Shuffles filtered dataset globally
        6. Initializes RydbergDataset which triggers processing/caching
    
    Returns:
        RydbergDataset: Fully processed dataset ready for model training.
    
    Raises:
        FileNotFoundError: If any file in CONFIG['data_paths'] does not exist.
        Exception: Any error during Parquet reading, concatenation, or sampling.
    
    """
    df_list = []
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        table = pq.read_table(path)
        df_temp = table.to_pandas()
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    
    # Apply initial size-based filtering
    filtered_data = []
    for ny, max_points in CONFIG['max_points_per_ny'].items():
        ny_data = df[df['Ny'] == ny]
        if len(ny_data) > max_points:
            ny_data = ny_data.sample(max_points, random_state=CONFIG['random_seed'])
        filtered_data.append(ny_data)
    
    df_filtered = pd.concat(filtered_data, ignore_index=True)
    df_shuffled = df_filtered.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    return RydbergDataset(dataframe=df_shuffled, root=CONFIG['processed_dir'])


def main() -> None:
    """
    Main entry point for dataset processing pipeline.
    
    Execution sequence:
        1. Sets dynamic configuration values (CPU worker count)
        2. Configures logging output format and level
        3. Sets random seeds for reproducibility
        4. Configures PyTorch thread count for optimal CPU utilization
        5. Creates processed directory if missing
        6. Loads and processes dataset via load_data()
        7. Logs summary statistics and sample data inspection
    
    
    Raises:
        Exception: Any unhandled exception during dataset processing.
    """
    set_dynamic_config()
    setup_logging()
    set_seed(CONFIG['random_seed'])
    
    # Optimize PyTorch CPU thread usage
    torch.set_num_threads(CONFIG['num_workers'])
    
    try:
        # Ensure output directory exists
        os.makedirs(CONFIG['processed_dir'], exist_ok=True)
        
        # Process dataset (triggers computation if not cached)
        dataset = load_data()
        logging.info(f"Finished processing. Dataset length: {len(dataset)}")
        
        # Log sample data for verification
        if len(dataset) > 0:
            sample = dataset[0]
            logging.info(f"Sample data object: {sample}")
            logging.info(f"Node feature dimensions: {sample.x.shape}")
            logging.info("Node features structure: [x_pos, y_pos, rydberg_prob, subsystem_mask]")
            logging.info(f"Edge index shape: {sample.edge_index.shape}")
            logging.info(f"Edge attribute dimensions: {sample.edge_attr.shape if sample.edge_attr is not None else 'N/A'}")
            logging.info(f"Target value (VNE): {sample.y.item():.6f}")
            
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise


if __name__ == "__main__":
    main()