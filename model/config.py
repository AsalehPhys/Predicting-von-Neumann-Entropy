FEATURE_SCHEMA = {
    'node_features': {
        'total_dim': 4,  # [x, y, rydberg_prob, subsystem_mask]
        'names': ['x_pos', 'y_pos', 'rydberg_prob', 'mask']
    },
    'edge_features': {
        'total_dim': 3,  # [angle, connected_corr, normalized_dist]
        'names': ['angle', 'connected_corr', 'dist_norm']
    }
}