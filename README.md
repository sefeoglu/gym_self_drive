# gym self drive
* The project uses the CarRacing environment of OpenAI.




# Abstract

*  Summarize the project and aims of it

# Motivation

* Give motivation and Examples from literature

# Neural Network Model
* Variational Auto Encoder

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
```gym_self_drive/model/VAE_RL/train_VAE.py``` :


#  Agent:
```gym_self_drive/model/VAE_RL/AgentTrain.py``` :

 * Explain reward mechanism
 * Explain observation and optimum action
 * Explain best params


# Results

Evaluate configurations of VAE model
* Rewards Graph of Config 1

![alt text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/rewards_1.png)

* Rewards Graph of Config 3

![alt_text](https://github.com/sefeoglu/gym_self_drive/blob/master/gym_self_drive/model/images/rewards3.png)

* Video

[![CarRacing-V0 ](https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/CarRacing-v0/poster.jpg)](https://www.youtube.com/watch?v=76iVVovnxhw)
# Data 
[CarRacing-V0 Data](https://drive.google.com/drive/folders/1mZk_yBLN-Iak_E8ewBJSl0rz1W1ckszM?usp=sharing)

# Saved Models :
[VAE Models](https://drive.google.com/drive/folders/1mZk_yBLN-Iak_E8ewBJSl0rz1W1ckszM?usp=sharing)
