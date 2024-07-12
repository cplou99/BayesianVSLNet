# BayesianVSLNet

CVPR2024 Ego4D Step Grounding Challenge Winner
This code is built over two other repos (NaQ and ego4d-goalstep). From there, we build our approach BayesianVSLNet: Bayesian temporal-order priors for test time refinement. Please, review the arxiv version for further details.

## Install
'''
git clone BayesianVSLNet
conda env create -f environment.yml
'''

## Video Features
We use both Omnivore-L and EgoVLPv2 video features. They should be located at ./ego4d-goalstep/step-grounding/data/features/
You can extract them or directly download them in the next link.

## Model 
It is necessary to locate the EgoVLPv2 weights to extract text features in BayesianVSLNet/NaQ/VSLNet_Bayesian/model/EgoVLP_weights. You can find a checkpoint of the complete model at .

### Train
cd ego4d-goalstep/step_grounding/
bash train_Bayesian.sh experiments/

### Inference
cd ego4d-goalstep/step_grounding/
bash ./ego4d-goalstep/step_grounding/infer_Bayesian.sh experiments/
