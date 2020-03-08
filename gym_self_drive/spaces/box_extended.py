'''

Sefika Efeoglu  sefeoglu@gmail.com
'''

from gym import spaces
import numpy as np


class BoxExtended(spaces.Box):
    def sample(self, sampling="Random"):

      if sampling == "Random":
        self.random_sampling()
      elif sampling == "Gibbs":
        self.gibbs_sampling()
      elif sampling == "Logic":
        self.logic_sampling()
      elif sampling == "Importance":
        self.importance_sampling()
        
      
    def gibbs_sampling(self):
      '''
      Improve gibbs sampling Algorithm
      '''
      pass
    def logic_sampling(self):
      '''
      Logic Sampling
      '''
      pass
    def importance_sampling(self):
      '''
        Importance Sampling
      '''
      pass
    def random_sampling(self):
      '''
      Random Sampling
      '''
      pass
     