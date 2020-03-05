import numpy as np

class EMA:
    def __init__(self, decay, size):
        self.decay = decay
        self.size = size
        self.value = None
    
    def apply(self, value):
        if self.value is None:
            self.value is value
        else:
            self.value = value * (1 - decay) + self.value * decay
        return self.value
    
    def get(self):
        if self.value is None:
            return np.ones(size)
        return self.value