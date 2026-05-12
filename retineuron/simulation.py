import numpy as np
import matplotlib.pyplot as plt
from neuron import h


class ExtracellularSimulation:
    def __init__(self, cell, electrode):
        self.cell = cell
        self.electrode = electrode

        self.t_field = np.array(electrode.t)
        self.t_play = self.t_field - self.t_field[0]

        self.Ve_seg = None
        self.ve_vectors = []

    def compute_segment_potentials(self):
        seg_xyz = self.cell.seg_xyz  # assumes that skel and vfield are already aligned, in same coords

        n_seg = seg_xyz.shape[0]
        n_t = len(self.t_field)

        xyz_rep = np.repeat(seg_xyz, n_t, axis=0)
        t_rep = np.tile(self.t_field, n_seg)

        query_points = np.column_stack([
            xyz_rep[:, 0],  # x
            xyz_rep[:, 1],  # y
            xyz_rep[:, 2],  # z
            t_rep,
        ])

        Ve_flat = self.electrode.interpolator(query_points)
        self.Ve_seg = Ve_flat.reshape(n_seg, n_t)

        return self.Ve_seg
    
    def apply_extracellular_potential(self):
        # Ve_seg -> seg._ref_e_extracellular
        pass

    def run(self):
        # run it
        pass

    def record(self):
        # record Vm, vext, terminal voltage, ...
        pass