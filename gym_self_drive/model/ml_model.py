'''
Authors: Sefika Efeoglu 
sefikaefeoglu@gmail.com
sefeoglu@github
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

class ML_model():
    def __init__(self):
        print("Sefika")

    def simple_model(self, obs_shape, nb_actions):
        '''
        obs_shape :  observation_shape
        nb_actions :  number of actions
        Return simple neural network model for q learning
        '''
        model = Sequential()
        model.add(Flatten(input_shape=(3,) + obs_shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())


        return model


    def deep_model(self):
        '''
        Improve deep neural network model
        return model
        '''