import scanpy as sc
import pandas as pd
import numpy as np

class DataLoader():
    
    def __init__(self):
        pass


    def load_data(self, path):
        print('\nLoading data from h5ad file.')

        self.adata = sc.read_h5ad(path)
        self.gene_expressions = pd.DataFrame.sparse.from_spmatrix(self.adata.X, columns=self.adata.var_names).astype(float).to_numpy()
        self.n_genes = self.gene_expressions.shape[1]

        print('\nSuccesfully loaded the data.')


    def get_gene_expressions_excluding_knockout(self, gene, n_instances=-1):
        # Get cell conditions
        condition = self.adata.obs['condition'].to_numpy()

        # Get cell indices
        indices = np.arange(len(condition))

        # Remove cell indices with gene knockout
        indices = indices[condition != gene]

        # Sample n_instances
        if n_instances != -1:
            indices = np.random.choice(indices, size=n_instances)

        # Select the gene expressions
        gene_expressions = self.gene_expressions[indices]

        return gene_expressions


    def get_gene_expressions_unperturbed_cells(self):
        # Get cell conditions
        condition = self.adata.obs['condition'].to_numpy()

        # Get cell indices
        indices = np.arange(len(condition))

        # Keep unperturbed cell indices only
        indices = indices[condition == 'Unperturbed']

        # Select the gene expressions
        gene_expressions = self.gene_expressions[indices]

        return gene_expressions

    
    def get_features_and_targets_excluding_knockout(self, gene, gene_idx, network_bool, n_instances):
        # Get data
        gene_expressions = self.get_gene_expressions_excluding_knockout(gene, n_instances=n_instances)

        # Get target
        target = gene_expressions[:,gene_idx]

        # Get features
        feature_indices = np.argwhere(network_bool[:,gene_idx] == 1)
        feature_indices = list(feature_indices.flatten())
        features = gene_expressions[:,feature_indices]

        return features, target


    def get_state_proportions_of_condition(self, condition):
        # Get states
        states = self.adata.obs['state'][self.adata.obs['condition'] == condition]
        
        # Calculate proportions
        mapping = {'progenitor': 0, 'effector': 1, 'terminal exhausted': 2, 'cycling': 3, 'other': 4}
        proportions = np.array([0,0,0,0,0])
        unique, counts = np.unique(states, return_counts=True)
        for idx, state in enumerate(unique):
            proportions[mapping[state]] = counts[idx]

        proportions = proportions / sum(counts)

        return proportions