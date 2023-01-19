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
run.py = run the final experiment

the folder "models" contains the infered GR network
the folder "networks" contains the models to recalculate gene expressions

#### Generate the output
just run "run.py" in cmd or in your IDE and the output will be saved to the "solution" folder