# BayesianVSLNet

CVPR2024 Ego4D Step Grounding Challenge Winner.

This code is built over two other repos (NaQ and ego4d-goalstep). From there, we build our approach BayesianVSLNet: Bayesian temporal-order priors for test time refinement. Please, review the arxiv version for further details.

SOON: We will release checkpoints and pre-extracted video features.

## Install
```ruby
git clone https://github.com/cplou99/BayesianVSLNet
conda env create -f environment.yml
conda activate goalstep
```

## Video Features
We use both Omnivore-L and EgoVLPv2 video features. They should be located at ./ego4d-goalstep/step-grounding/data/features/.

## Model 
It is necessary to locate the EgoVLPv2 weights to extract text features in BayesianVSLNet/NaQ/VSLNet_Bayesian/model/EgoVLP_weights.

### Train
```ruby
cd ego4d-goalstep/step_grounding/
bash train_Bayesian.sh experiments/
```

### Inference
```ruby
cd ego4d-goalstep/step_grounding/
bash infer_Bayesian.sh experiments/
```
