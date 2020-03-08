import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory

from rl.agents import SARSAAgent

from ml_model import ML_model
# environment name
ENV_NAME = 'gym_self_drive:drive-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = 3
ml_model_object = ML_model()
simple_ml_model = ml_model_object.simple_model(env.observation_space.shape,nb_actions)

memory = EpisodeParameterMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=simple_ml_model, nb_actions=nb_actions, policy=policy, memory=memory,
              nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)