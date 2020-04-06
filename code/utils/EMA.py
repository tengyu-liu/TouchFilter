import numpy as np

class EMA:
    def __init__(self, decay, size):
        self.decay = decay
        self.size = size
        self.value = None
    
    def apply(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = value * (1 - self.decay) + self.value * self.decay
        return self.value
    
    def get(self):
        if self.value is None:
            return np.ones(self.size)
        return self.value