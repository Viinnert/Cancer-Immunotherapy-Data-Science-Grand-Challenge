##### Dependencies
numpy=1.23.4
pandas=1.5.2
scanpy=1.9.1
tqdm=4.64.1
sklearn=1.2.0
pickle=4.0

python=3.9.13

#### files
Dataloader.py & Inference.py & Predictor.py = the classes doing the calculations
experiment.ipynb = just for playing around with the code
run_excl_infer&train.py = run the final experiment by loading in the inferred network and trained models.
run_incl_infer&train.py = run the final experiment starting from scratch. So including the network inference and model training. (Computationally expensive)

the folder "models" contains the infered GR network
the folder "networks" contains the models to recalculate gene expressions

#### Generate the output
just run "run_excl_infer&train.py" in cmd or in your IDE and the output will be saved to the "solution" folder.
or run "run_inccl_infer&train.py" in cmd or your IDE. This will also infer a new network and train new models and save them to their folders.

NOTE: Due to the size of the network and model's file i could not upload these to topcoder.

NOTE: Make sure cmd is inside the Challenge1/code folder. Loading/saving networks/models is relative to this folder. 
To refer to the correct file for loading the dataset, you can change the "adata_path" according to where you save the dataset.