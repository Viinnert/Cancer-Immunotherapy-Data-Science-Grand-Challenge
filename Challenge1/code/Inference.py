import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

class Inference():

    def __init__(self, dataloader):
        self.dataloader = dataloader

        try:
            # network[i,j] => effect gene i on gene j 
            self.n_genes = self.dataloader.n_genes
            self.network = np.zeros((self.n_genes, self.n_genes))
            self.network_bool = np.zeros((self.n_genes, self.n_genes))
        except:
            raise AttributeError('Inference initialized but dataLoader has not yet loaded any data.')


    def infer(self, n_instances, n_estimators, max_depth):
        print('\nInfering genetic regulatory network.')

        # Get list of genes
        genes = self.dataloader.adata.var_names
        
        # Get importances
        for gene_idx, gene in enumerate(tqdm(genes)):
            self.get_importances(gene, gene_idx, n_instances, n_estimators, max_depth)

        # Ranking the genes
        self.order_genes()

        print('\nFinished infering genetic regulatory network.')
        

    def get_importances(self, gene, gene_idx, n_instances, n_estimators, max_depth,):
        'Calculates the importance of all genes on a given gene'
        # Get data
        gene_expressions = self.dataloader.get_gene_expressions_excluding_knockout(gene, n_instances)

        # Split data in features and target
        target = gene_expressions[:,gene_idx]
        features = np.delete(gene_expressions, gene_idx, axis=1)

        # Get importances
        random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        #gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'n_jobs': -1}
        #random_forest = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, **gpu_params)
        random_forest.fit(features, target)
        importances = random_forest.feature_importances_

        # Save to network
        importances = np.insert(importances, gene_idx, 0)
        self.network[:,gene_idx] = importances

    
    def order_genes(self):
        print('\nDetermining gene order.')
        importances = np.sum(self.network, axis=1)
        _, inverse = np.unique(importances, return_inverse=True)

        # From important to not important
        inverse = list(reversed(inverse))

        # Remove upward edges
        for idx, gene_idx in enumerate(inverse):
            down_genes = inverse[idx:]
            self.network[down_genes,gene_idx] = 0


    def determine_directionality(self):
        print('Determining directionalty.')
        for i in tqdm(range(self.n_genes)):
            for j in range(i):
                if self.network[i,j] > self.network[j,i]:
                    self.network[j,i] = 0
                else:
                    self.network[i,j] = 0


    def find_lowest_threshold(self, epsilon=0.1):
        print('\nFinding importance threshold for which no cycles occur.')
        low_thresh = 0
        high_thresh = 1 

        while high_thresh - low_thresh > epsilon:
            thresh = (high_thresh + low_thresh) / 2
            self.set_importance_threshold(thresh)
            cycle = self.check_cycle()

            if cycle == True:
                low_thresh = thresh
            else:
                high_thresh = thresh

        self.set_importance_threshold(high_thresh)
        print(f'The importance threshold for which no cycles occur lies between {low_thresh} and {high_thresh}.')
        print(f'Importance threshold set to {high_thresh}.')

        return high_thresh


    def check_cycle(self):
        print(f'\nChecking for cycles with importance threshold {self.importance_threshold}')
        visited = [False] * self.n_genes
        recstack = [False] * self.n_genes

        for gene in range(self.n_genes):
            if not visited[gene]:
                if self.check_cycle_util(gene, visited, recstack):
                    print('Cycle found')
                    return True
        print('No cycle found')
        return False


    def check_cycle_util(self, gene, visited, recstack):
        visited[gene] = True
        recstack[gene] = True
        
        linked_genes = np.argwhere(self.network_bool[gene] == 1)
        linked_genes = list(linked_genes.flatten())

        for linked_gene in linked_genes:
            if not visited[linked_gene]:
                if self.check_cycle_util(linked_gene, visited, recstack):
                    return True
            elif recstack[linked_gene]:
                return True

        recstack[gene] = False
        return False


    def check_cycle2(self):
        #graph = { 1: [2, 3, 5], 2: [1], 3: [1], 4: [2], 5: [2] }
        graph = {}
        for gene_idx in range(self.n_genes):
            graph[gene_idx] = list(np.argwhere(self.network_bool[gene_idx] == 1).flatten())

        #cycles = [[node]+path  for node in graph for path in self.dfs(graph, node, node)]
        cycles = []
        for node in graph:
            for path in self.dfs(graph, node, node):
                cycle = [node]+path
                print(cycle)
                cycles.append(cycle)

        return cycles


    def dfs(self, graph, start, end):
        fringe = [(start, [])]
        while fringe:
            state, path = fringe.pop()
            if path and state == end:
                yield path
                continue
            for next_state in graph[state]:
                if next_state in path:
                    continue
                fringe.append((next_state, path+[next_state]))


    def get_knockout_order(self, gene):
        # Get list of genes
        genes = self.dataloader.adata.var_names.to_numpy()

        # Get index of knockout gene 
        ko_idx = np.argwhere(genes == gene)[0][0]

        # Create and extend order list
        order = [ko_idx]
        
        # Walk through path
        for gene_idx in order:
            new_genes_idxs = np.argwhere(self.network_bool[gene_idx] == 1)

            # Add new genes
            new_genes_idxs = list(new_genes_idxs.flatten())
            order.extend(new_genes_idxs)

        # drop duplicates except for the final one
        order = list(reversed(order))
        order = pd.unique(order)
        order = list(reversed(order))

        return order


    def check_for_incoming_edges(self, gene_idx):
        # Find all incoming edges
        incoming_edges = np.argwhere(self.network_bool[:,gene_idx] == 1)
        incoming_edges = list(incoming_edges.flatten())

        if len(incoming_edges) == 0:
            return False
        elif len(incoming_edges) > 0:
            return True


    def set_importance_threshold(self, importance_threshold):
        self.importance_threshold = importance_threshold
        self.network_bool = np.where(self.network > importance_threshold, 1, 0)
    

    def save_network(self, network_name):
        np.save(f'networks/{network_name}', self.network)


    def load_network(self, network_name):
        print('\nLoading genetic regulatory network.')
        self.network = np.load(f'networks/{network_name}.npy')