from gym import spaces
import numpy as np
class BoxExtended(spaces.Box):
    def __init__(self, low, high, shape=None, dtype=np.float32):
      spaces.Box(low, high, shape=None, dtype=np.float32)
    def sample(self):
        #TODO#
        # Develop this part with new sampling algorithms
        return 0