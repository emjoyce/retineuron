import numpy as np
from neuron import h
import pickle
from scipy.interpolate import RegularGridInterpolator

# will contain:
# Electrode
# Electrode layouts
# Electric field
# stimulus protocols (ie anodic first biphastic etc)

class Electrode:
    def __init__(self, dir):

        with open(dir, "rb") as f: # open and read the pickle file (in post process file)
            data = pickle.load(f)
        
        self.V = data['v(x,y,z,t)_mv']                      # voltage: v(x position, y position, z position, time point) shape(501, 501, 3, 97)
        self.x, self.y = data['2d_mesh_um']                 # x/y 2Dmesh (um)
        self.z = np.array(data['z_um'])                     # z coordinates in micrometers
        self.t_start = np.array(data['t_start_ms'])         # starting time (ms)
        self.t_end = np.array(data['t_end_ms'])             # ending time (ms)
        self.pixel_coords = data['pixel_coordinates_um']    # 
    
        # use midpoint of each time bin
        self.t = (self.t_start + self.t_end) / 2

        # extract 1D x and y coordinates from meshgrid
        self.x = self.x[0, :]
        self.y = self.y[:, 0]

        self.interpolator = RegularGridInterpolator(
            (self.x, self.y, self.z, self.t),
            self.V,
            bounds_error=False,
            fill_value=None)

    def report_range(self):
        print("x range:", self.x.min(), self.x.max())
        print("y range:", self.y.min(), self.y.max())
        print("z range:", self.z.min(), self.z.max())
        print("t range:", self.t.min(), self.t.max())
        print("V min/max:", np.nanmin(self.V), np.nanmax(self.V))
        print("V shape:", self.V.shape)


    def pull_xyzt(self, x, y, z, t): # what is the unit of 
        """
        Find electrode field/value given x,y,z,t. location
        """
        point = np.array([x, y, z, t]) 
        return self.interpolator(point).item()

    def placing_elec(self, x, y, z, t, electrode_pos):
        """
        Neuron-centered coordinate system.

        x,y,z = point in neuron-centered world space
        electrode_pos = where electrode is located relative to neuron
        """

        ex, ey, ez = electrode_pos

        # convert world coordinate to electrode-centered local coordinate
        x_local = x - ex
        y_local = y - ey
        z_local = z - ez

        point = np.array([x_local, y_local, z_local, t])
        return self.interpolator(point).item()