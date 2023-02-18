import pandas as pd

from DataLoader import DataLoader
from Inference import Inference
from Predictor import Predictor

adata_path = '../../Data/sc_training.h5ad'

n_infer_instances = 1000 # Number of instances used for infering the network. -1 = all 
n_infer_estimators = 4 # Number of trees in random forest for infering the network
max_depth_infer = 10 # max depth of random forest for infering the network
importance_threshold = 0.011

n_train_instances = 1000 # Number of instances used training predictor models. -1 = all 
n_train_estimators = 4 # Number of trees in random forest training predictor models
max_depth_train = 10 # max depth of random forest training predictor models

n_features = 1000 # The number of most important genes to use for cell state classifications
n_components = 100 # Number of principal components
n_neighbors = 11 # Number of neighbors for the k-NN state classifier


# Dataloader
dataloader = DataLoader()
dataloader.load_data(adata_path)

# Inference
inference = Inference(dataloader)
inference.load_network('GRNetwork')
inference.set_importance_threshold(importance_threshold)

# Predictor
predictor = Predictor(dataloader, inference)
predictor.load_models('GRNetwork_models')
predictor.fit(n_features, n_components, n_neighbors)

states = ['progenitor', 'effector', 'terminal exhausted', 'cycling', 'other']

# Predict validation set
val_genes = ['Aqr', 'Bach2', 'Bhlhe40']
val_predictions = pd.DataFrame(0, index=val_genes, columns=states)
for idx, gene in enumerate(val_genes):
    predictions = predictor.predict_knockout_effect(gene, n_components, n_neighbors)
    val_predictions.iloc[idx] = predictions

# Predict test set
test_genes = ['Ets1', 'Fosb', 'Mafk', 'Stat3']
test_predictions = pd.DataFrame(0, index=test_genes, columns=states)
for idx, gene in enumerate(test_genes):
    predictions = predictor.predict_knockout_effect(gene, n_components, n_neighbors)
    test_predictions.iloc[idx] = predictions

# Save predictions
val_predictions.to_csv('../solution/validation_output.csv')
test_predictions.to_csv('../solution/test_output.csv')