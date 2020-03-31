# gym self drive
* The project uses the CarRacing environment of OpenAI.

# Abstract

*  Summarize the project and aims of it


# Neural Network Model

* Variational Auto Encoder :

LaserScanner of Car providers observation regarding the envirement and then it sends to VAE. RL Agent makes a decision from the observation. Agent gets -1,+1 Rewards from this action. 

![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/VAEModel.png)

## Functionalities

```gym_self_drive/model/VAE_RL/model.py``` :



* First Configuration of VAE
  ![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/Config1.png)

* ``Second Configuration of VAE`` : Work in progress 

    64 32 64 128 64  
* Third Configuration of VAE
   ![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/Config2.png)

* ```Forth Configuration of VAE```: Work in progress

     64 128 256 512
# Train VAE
```gym_self_drive/model/VAE_RL/train_VAE.py``` : Train VAE with generated data from CarRacing-V0 environment.


#  Agent:
```gym_self_drive/model/VAE_RL/AgentTrain.py``` : Train agent with VAE and RL. For VAE parameters, CMAEvolutionStrategy package is used. It provider parameter for VAE, so hyperparameter tuning is not needed for this project.

 * Explain reward mechanism : The CarRacing finishes when Car obtains 9999 rewards according the setup of CarRacing-v0 Environment.
 
 * Explain observation and optimum action
 
 * Explain best params refers to CMA parameter when Agent got high rewards while training it.


# Results

Evaluate configurations of VAE model
* Rewards Graph of Config 1

![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/rewards_1.png)

* Rewards Graph of Config 3

![alt_text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/rewards3.png)

* Many Iteration with config 1

![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/For4Agent_reward.png)

* Video

[![CarRacing-V0 ](https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/CarRacing-v0/poster.jpg)](https://www.youtube.com/watch?v=76iVVovnxhw)
# Data 
[CarRacing-V0 Data](https://drive.google.com/drive/folders/1mZk_yBLN-Iak_E8ewBJSl0rz1W1ckszM?usp=sharing)

# Saved Models :
[VAE Models](https://drive.google.com/drive/folders/1mZk_yBLN-Iak_E8ewBJSl0rz1W1ckszM?usp=sharing)
