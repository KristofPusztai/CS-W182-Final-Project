# UC Berkeley CS W182 RL Project Readme

# Introduction

This assignment was made to be run on Google Colab, due to the port process we did in our code to leverage the GPU enviornment. The files should be in one zipped folder, due to the directory design, but we will include multiple subfolders for easier access.

# Contents
We have five different subfolders, four of which are for training the baseline models. (50FruitBot, 100FruitBot, 250FruitBot, and 500FruitBot). The PP02+Decoupled folder contains the training notebooks for our novel decoupled model. The evaluation folder contains the notebooks that we used to evaluate our model. For each model and for each number of train levels, we have both an initial notebook(that creates and trains the model until it reaches reward= 10) and a continued notebook(loads the model and trains it up till reward=20).
  submissions.zip  
        network.py
          This file is where we implemented the layers that are used for the  different model architecture we studied in this project
        helpers.py
          This file contains functions that helped us visualize our training process. It includes functions from the OpenAI gym library that recorded the agent playing Fruitbot.
        ImpalaCNN.py
          This is the implementation of ImpalaCNN used for training policy and value functions in baseline models ***
        mixreg.py
          Implementation of the mixreg observation mixing algroithm
        MIXREG_ImpalaCnn.py
          ImpalaCNN implementation when using mixreg based model
        DECOUPLED_ImpalaCnn.py
          ImpalaCNN implementation using decoupling, used to train the policy and value functions for our novel model

# CSV file
We evaluated the model in the test enviornment for 20,000 train time steps. 
# PDF Writeup
https://github.com/KristofPusztai/CSW182-Final/blob/master/Reinforcement_Learning_Generalization_via_Partially_Decoupled_Policy-Value_Networks.pdf
