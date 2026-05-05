import numpy as np
from neuron import h

# will contain:
# Electrode
# Electrode layouts
# Electric field
# stimulus protocols (ie anodic first biphastic etc)

# TODO: should active and return be separate?
class Electrode:
    def __init__(self, center_pos, diameter, shape='hex', role ='active',
                height = 0):
        self.position = np.array(center_pos)
        self.diameter = diameter
        self.shape = shape
        self.role = role
        self.height = height
    pass 