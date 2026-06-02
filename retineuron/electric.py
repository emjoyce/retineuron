import numpy as np
from neuron import h
import pickle
from scipy.interpolate import RegularGridInterpolator


# TODO: might be y x z t if it is stored as i,j
class Electrode:
    def __init__(self, dir):

        with open(dir, "rb") as f: # open and read the pickle file (in post process file)
            data = pickle.load(f)
        
        self.V = data['v(x,y,z,t)_mv']                      # voltage: v(x position, y position, z position, time point) 
        self.x, self.y = data['2d_mesh_um']                 # x/y 2Dmesh (um)
        self.z = np.array(data['z_um'])                     # z coordinates in microns
        self.t_start = np.array(data['t_start_ms'])         # starting time (ms)
        self.t_end = np.array(data['t_end_ms'])             # ending time (ms)
        self.pixel_coords = data['pixel_coordinates_um']    # pixel coordinates in microns (x,y,z) for each point in the 3D grid
    
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

    def shifted_field_withSpace(self, X, Y, Z, T, shift=(0, 0, 0)): # in um
        """
        X,Y,Z helps to create a space 
        T: field in which time_point to be shifted
        Move the electric field around in 3D space.
        shift(x, y, z) = the distance you want to move the default center of the electric field.
        """
        shift_x, shift_y, shift_z = shift

        points = np.column_stack([ # create shift version
            X.ravel() - shift_x,
            Y.ravel() - shift_y,
            Z.ravel() - shift_z,
            np.full(X.size, T)
        ])

        V = self.interpolator(points)
        return V.reshape(X.shape) # need for showing the plot
    
    def shifted_field(self, x, y, z, t, shift=(0, 0, 0)): 
        """
        only shift the field mathematically 
        shift(x, y, z) = the distance you want to move the default center of the electric field.
        """
        sx, sy, sz = shift

        return self.interpolator([
            x - sx,
            y - sy,
            z - sz,
            t
        ])