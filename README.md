# BayesianVSLNet - Ego4D Step Grounding Challenge CVPR24 :trophy: 
:soon:: We will release checkpoints and pre-extracted video features.

[[ArXiv]](https://arxiv.org/abs/2406.09575) [[Leaderboard]](https://eval.ai/web/challenges/challenge-page/2188/leaderboard/5405)
# Challenge
The challenge is built over [Ego4d-GoalStep](https://github.com/facebookresearch/ego4d-goalstep?tab=readme-ov-file) dataset and code.

**Goal:** Given an untrimmed egocentric video, identify the temporal action segment corresponding to a natural language description of the step. Specifically, predict the (start_time, end_time) for a given keystep description.
 
 <div align="center">
 <img src="images/teaser_step_grounding.drawio.png" alt="Challenge" width="700"/>
</div>

You will find in the [leaderboard](https://eval.ai/web/challenges/challenge-page/2188/leaderboard/5405) :rocket: the results in the test set for the best approaches. Our method is currently in the first place :rocket::fire:.

 # BayesianVSLNet
We build our approach BayesianVSLNet: Bayesian temporal-order priors for test time refinement. Our model significantly improves upon traditional models by incorporating a novel Bayesian temporal-order prior during inference, which adjusts for cyclic and repetitive actions within video, enhancing the accuracy of moment predictions. Please, review the [paper](https://arxiv.org/abs/2406.09575) for further details.

![Alt text](images/Model.png)


## Install
```ruby
git clone https://github.com/cplou99/BayesianVSLNet
pip install -r requirements.txt
```

## Video Features
We use both Omnivore-L and EgoVLPv2 video features. They should be pre-extracted and located at ./ego4d-goalstep/step-grounding/data/features/.

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
