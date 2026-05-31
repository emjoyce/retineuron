import numpy as np
import matplotlib.pyplot as plt
from neuron import h


class ExtracellularSimulation:
    def __init__(self, cell, electrode):
        self.cell = cell
        self.electrode = electrode

        self.t_field = np.array(electrode.t)
        self.t_play = self.t_field - self.t_field[0]

        self.v_external_seg = None
        self.ve_vectors = []

    def compute_segment_potentials(self):
        seg_xyz = self.cell.seg_xyz  # assumes that skel and vfield are already aligned, in same coords

        n_seg = seg_xyz.shape[0]
        n_t = len(self.t_field)

        xyz_rep = np.repeat(seg_xyz, n_t, axis=0)
        t_rep = np.tile(self.t_field, n_seg)

        # run them all at once for speed 
        query_points = np.column_stack([
            xyz_rep[:, 0],  # x
            xyz_rep[:, 1],  # y
            xyz_rep[:, 2],  # z
            t_rep,
        ])

        v_external_flat = self.electrode.interpolator(query_points)
        self.v_external_seg = v_external_flat.reshape(n_seg, n_t)

        return self.v_external_seg
    
    def apply_extracellular_potential(self):
        """
        apply the electrode voltage over time into every NEURON compartment

        Uses:
            self.v_external_seg[segment, time] in mV
            self.electrode.t in ms
        """

        # if we haven't already computed the potentials at each segment, do it
        if self.v_external_seg is None:
            self.compute_segment_potentials()

        # Shift time so the NEURON simulation starts at t = 0 
        self.t_field = np.array(self.electrode.t)
        self.t_play = self.t_field - self.t_field[0]

        # NEURON vector containing the time points 
        self.t_vec = h.Vector(self.t_play)

        self.ve_vectors = []

        # Play one extracellular voltage over time into each segment.
        for seg_idx, seg in enumerate(self.cell.seg_refs):
            ve_vec = h.Vector(self.v_external_seg[seg_idx, :])
            ve_vec.play(seg._ref_e_extracellular, self.t_vec, True)
            self.ve_vectors.append(ve_vec)

    def run(self, dt=0.025, v_init=None, record_segments=None):
        """
        Apply extracellular stimulation, record response, and run NEURON.

        dt: NEURON integration time step in ms
        v_init: initial membrane potential in mV
        record_segments: optional list of segment indices to record
        """

        # Use the cell's passive reversal as default initialization voltage.
        if v_init is None:
            v_init = self.cell.e_pas

        # Put the RPSim extracellular potential into NEURON.
        self.apply_extracellular_potential()

        # Set up recordings before running.
        self.record(seg_indices=record_segments)

        # Run the simulation.
        h.dt = dt
        h.tstop = float(self.t_play[-1])

        h.finitialize(v_init)
        h.continuerun(h.tstop)

        # Convert recordings to arrays and store them.
        self.results = {
            "t_ms": np.array(self.t_rec),
            "v_mV": np.array([np.array(v) for v in self.v_rec]),
            "vext_mV": np.array([np.array(vext) for vext in self.vext_rec]),
            "eext_mV": np.array([np.array(eext) for eext in self.eext_rec]),
            "recorded_seg_indices": np.array(self.recorded_seg_indices),
            "seg_xyz_um": self.cell.seg_xyz,
            "Ve_input_mV": self.v_external_seg,
        }

        return self.results
    
    def record(self, seg_indices=None):
        """
        Set up recordings before running the simulation.

        Records:
            time
            membrane voltage v
            extracellular voltage vext[0]
            imposed extracellular source e_extracellular
        """

        # Default: record every segment.
        if seg_indices is None:
            seg_indices = range(len(self.cell.seg_refs))

        self.recorded_seg_indices = list(seg_indices)

        # Record simulation time.
        self.t_rec = h.Vector().record(h._ref_t)

        # Store one Vector per recorded segment.
        self.v_rec = []
        self.vext_rec = []
        self.eext_rec = []

        for seg_idx in self.recorded_seg_indices:
            seg = self.cell.seg_refs[seg_idx]

            self.v_rec.append(h.Vector().record(seg._ref_v))
            self.vext_rec.append(h.Vector().record(seg._ref_vext[0]))
            self.eext_rec.append(h.Vector().record(seg._ref_e_extracellular))