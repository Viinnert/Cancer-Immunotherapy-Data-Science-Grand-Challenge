import numpy as np
import pickle 

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class Predictor():

    def __init__(self, dataloader, inference):
        self.dataloader = dataloader
        self.inference = inference
        self.fitted = False


    def train(self, n_instances, n_estimators, max_depth):
        self.models = {}

        genes = self.dataloader.adata.var_names
        print('\nTraining models for predicting gene expressions.')
        for gene_idx, gene in enumerate(tqdm(genes)):
            # Check for incoming edges
            has_incoming = self.inference.check_for_incoming_edges(gene_idx)

            if has_incoming:
                # Train model
                model = self.train_model(gene, gene_idx, n_instances, n_estimators, max_depth)

                # Save model
                self.models[gene] = model


    def train_model(self, gene, gene_idx, n_instances, n_estimators, max_depth):
        'Train a model for a single gene'
        # Get data
        features, target = self.dataloader.get_features_and_targets_excluding_knockout(gene, gene_idx, self.inference.network_bool, n_instances)

        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        model.fit(features, target)

        return model


    def predict_knockout_effect(self, gene, n_components, n_neighbors):
        # Check if fitted
        if self.fitted == False:
            raise RuntimeError('Predictor has nog yet been fitted.')

        print(f'\nPredicting state proportions after knockout of {gene}.')
        # Get data of unperturbed cells
        gene_expressions = self.dataloader.get_gene_expressions_unperturbed_cells()

        # Get order of genes
        order = self.inference.get_knockout_order(gene)

        # Introduce knockout
        gene_expressions[:,order[0]] = 0

        for gene_idx in order[1:]:
            gene_expressions = self.recalculate(gene_idx, gene_expressions)

        # Get proportions
        proportions = self.get_state_proportions(gene_expressions)
        
        return proportions


    def recalculate(self, gene_idx, gene_expressions):
        'Recalculates the gene expression of a given gene given a dataset of gene expressions'
        # Get features
        feature_indices = np.argwhere(self.inference.network_bool[:,gene_idx] == 1)
        feature_indices = list(feature_indices.flatten())
        features = gene_expressions[:,feature_indices]

        # Get model
        gene = self.dataloader.adata.var_names[gene_idx]
        model = self.models[gene]

        # Recalculate
        gene_expressions[:,gene_idx] = model.predict(features)

        return gene_expressions

    
    def get_state_proportions(self, predicted_expressions):
        # Check if fitted
        if self.fitted == False:
            raise RuntimeError('Predictor has nog yet been fitted.')

        # Do pca and nearest neighbor on predictions
        predicted_reduced = self.pca.transform(predicted_expressions[:,self.state_features_idxs])
        predicted_states = self.nearest_neighbors.predict(predicted_reduced)

        # Calculate proportions
        mapping = {'progenitor': 0, 'effector': 1, 'terminal exhausted': 2, 'cycling': 3, 'other': 4}
        proportions = np.array([0,0,0,0,0])
        unique, counts = np.unique(predicted_states, return_counts=True)
        for idx, state in enumerate(unique):
            proportions[mapping[state]] = counts[idx]

        proportions = proportions / sum(counts)

        return proportions

    
    def fit(self, n_features, n_components, n_neighbors):
        print('\nFitting predictor.')
        # Get gene expressions and states
        expressions = self.dataloader.gene_expressions
        states = self.dataloader.adata.obs['state'].to_numpy()

        # Feature selection
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        model.fit(expressions, states)
        importances = model.feature_importances_

        sorted_ = np.array(list(reversed(sorted(importances))))
        self.state_features_idxs = np.argwhere(importances > sorted_[n_features]).flatten()
    
        # Pca
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(expressions[:,self.state_features_idxs])

        # Nearest neighbor
        self.nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.nearest_neighbors.fit(reduced, states)

        self.fitted = True


    def save_models(self, model_name):
        with open(f'models/{model_name}.pickle', 'wb') as handle:
            pickle.dump(self.models, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_models(self, model_name):
        print('\nLoading models.')
        with open(f'models/{model_name}.pickle', 'rb') as handle:
            self.models = pickle.load(handle)