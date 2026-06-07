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

    def get_field_dt_ms(self):
        if self.t_play.size < 2:
            raise ValueError("Electrode time grid must contain at least two points to infer dt")

        dt_values = np.diff(self.t_play)
        dt_ms = float(dt_values[0])
        return dt_ms

    def compute_segment_potentials(self, position_offset_um=None):
        seg_xyz = self.cell.seg_xyz # assumes that skel and vfield are already aligned, in same coords

        if position_offset_um is not None:
            position_offset_um = np.asarray(position_offset_um, dtype=float)
            if position_offset_um.shape != (3,):
                raise ValueError("position_offset_um must be in 3d (x, y and z offset)")
            seg_xyz = seg_xyz + position_offset_um

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
        self.sampled_seg_xyz_um = seg_xyz
        self.sampled_seg_xyz_electrode_um = seg_xyz.copy()

        return self.v_external_seg
    
    def apply_extracellular_potential(self, position_offset_um=None):
        """
        apply the electrode voltage over time into every NEURON compartment

        Uses:
            self.v_external_seg[segment, time] in mV
            self.electrode.t in ms
        """

        # if we haven't already computed the potentials at each segment, do it
        if self.v_external_seg is None:
            self.compute_segment_potentials(position_offset_um=position_offset_um)

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
# TODO dt should be pulled from the e field 
    def run(self, dt = None, v_init=None, record_segments=None, position_offset_um=None):
        """
        Apply extracellular stimulation, record response, and run NEURON.

        dt: NEURON integration time step in ms. If None, infer from the
            electrode time grid.
        v_init: initial membrane potential in mV
        record_segments: optional list of segment indices to record
        position_offset_um: optional placement offset applied only when sampling
            the extracellular field
        """

        # Use the cell's passive reversal as default initialization voltage.
        if v_init is None:
            v_init = self.cell.e_pas

        # Put the RPSim extracellular potential into NEURON.
        self.apply_extracellular_potential(position_offset_um=position_offset_um)

        # Set up recordings before running.
        self.record(seg_indices=record_segments)

        # Run the simulation.
        if dt is None:
            dt = self.get_field_dt_ms()

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
            "sampled_seg_xyz_um": self.sampled_seg_xyz_um,
            "sampled_seg_xyz_electrode_um": self.sampled_seg_xyz_electrode_um,
            "Ve_input_mV": self.v_external_seg,
            "dt_ms": float(dt),
            "position_offset_um": np.zeros(3) if position_offset_um is None else np.asarray(position_offset_um, dtype=float),
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


class PlacementSweep:
    def __init__(self, cell, electrode):
        self.cell = cell
        self.electrode = electrode

    @staticmethod
    def get_placement_offsets_from_results(sweep_results):
        if "placement_offsets_um" in sweep_results:
            placement_offsets_um = np.asarray(sweep_results["placement_offsets_um"], dtype=float)
            if placement_offsets_um.ndim != 2 or placement_offsets_um.shape[1] != 3:
                raise ValueError("sweep_results['placement_offsets_um'] must have shape (n_points, 3)")
            return placement_offsets_um

        placement_anchor_xyz_um = np.asarray(sweep_results["placement_anchor_xyz_um"], dtype=float)
        native_anchor_xyz_um = np.asarray(sweep_results["native_anchor_xyz_um"], dtype=float)

        if placement_anchor_xyz_um.ndim != 2 or placement_anchor_xyz_um.shape[1] != 3:
            raise ValueError("sweep_results['placement_anchor_xyz_um'] must have shape (n_points, 3)")
        if native_anchor_xyz_um.shape != (3,):
            raise ValueError("sweep_results['native_anchor_xyz_um'] must have shape (3,)")

        return placement_anchor_xyz_um - native_anchor_xyz_um

    def _get_baseline_offset_um(self, center_um, placement_by="soma", dendrite_um=10.0, axis="z"):
        anchor_xyz_um = self.cell.get_placement_anchor_xyz(
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis=axis,
        )
        center_cell_um = np.asarray(center_um, dtype=float).copy()

        # Align the selected anchor in x/y, but keep the lowest z as the lowest z in soma or dend.
        return np.array([
            center_cell_um[0] - anchor_xyz_um[0],
            center_cell_um[1] - anchor_xyz_um[1],
            center_cell_um[2],
        ])

    def _run_offsets(self, placement_offsets_um, output_shape, terminal_um=10.0,
                     placement_by="soma", dendrite_um=10.0, axis="z",
                     dt=None, v_init=None):
        terminal_indices = self.cell.get_terminal_segment_indices(terminal_um=terminal_um)
        native_anchor_xyz_um = self.cell.get_placement_anchor_xyz(
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis=axis,
        )
        placement_anchor_xyz_um = placement_offsets_um + native_anchor_xyz_um

        traces = []
        t_ms = None

        for placement_offset_um in placement_offsets_um:
            sim = ExtracellularSimulation(self.cell, self.electrode)
            result = sim.run(
                dt=dt,
                v_init=v_init,
                record_segments=terminal_indices.tolist(),
                position_offset_um=placement_offset_um,
            )

            terminal_segment_traces = result["v_mV"].copy()
            terminal_trace = np.mean(terminal_segment_traces, axis=0)
            traces.append(terminal_trace)
            if t_ms is None:
                t_ms = result["t_ms"].copy()

        return {
            "placement_anchor_xyz_um": placement_anchor_xyz_um,
            "native_anchor_xyz_um": native_anchor_xyz_um,
            "placement_by": placement_by,
            "dendrite_um": float(dendrite_um),
            "placement_axis": axis,
            "t_ms": t_ms if t_ms is not None else np.array([]),
            "terminal_v_mV": np.array(traces).reshape(*output_shape, -1),
        }

    def build_grid_offsets(self, radius_um, spacing_um):
        # grid for each cell location to be tested
        radius_um = float(radius_um)
        spacing_um = float(spacing_um)

        if radius_um < 0:
            raise ValueError("radius_um must be greater than or equal to 0")
        if spacing_um <= 0:
            raise ValueError("spacing_um must be positive")

        xy_values = np.arange(-radius_um, radius_um + 0.5 * spacing_um, spacing_um)
        z_values = np.arange(0.0, radius_um + 0.5 * spacing_um, spacing_um)
        # keep z positive to only shift above the electrode, not under it! 
        grid_3d = np.meshgrid(xy_values, xy_values, z_values, indexing="ij")

        return np.column_stack([axis.ravel() for axis in grid_3d])

    def build_vertical_sheet_offsets(self, y_radius_um, z_radius_um, y_spacing_um, z_spacing_um):
        y_radius_um = float(y_radius_um)
        z_radius_um = float(z_radius_um)
        y_spacing_um = float(y_spacing_um)
        z_spacing_um = float(z_spacing_um)

        if y_radius_um < 0 or z_radius_um < 0:
            raise ValueError("y_radius_um and z_radius_um must be greater than or equal to 0")
        if y_spacing_um <= 0 or z_spacing_um <= 0:
            raise ValueError("y_spacing_um and z_spacing_um must be positive")

        x_values = np.array([0.0])
        y_values = np.arange(-y_radius_um, y_radius_um + 0.5 * y_spacing_um, y_spacing_um)
        z_values = np.arange(0.0, z_radius_um + 0.5 * z_spacing_um, z_spacing_um)
        sheet = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        return np.column_stack([axis.ravel() for axis in sheet])

    def run_centered_grid(self, center_um=(0.0, 0.0, 0.0), radius_um=0.0, spacing_um=10.0,
                          terminal_um=10.0, placement_by="soma", dendrite_um=10.0,
                          dt=None, v_init=None):
        # run cell sim at multiple placements in the electric field and pull out terminal responses
        center_um = np.asarray(center_um, dtype=float)
        if center_um.shape != (3,):
            raise ValueError("center_um must be length 3")

        radius_um = float(radius_um)
        spacing_um = float(spacing_um)
        x_values = np.arange(-radius_um, radius_um + 0.5 * spacing_um, spacing_um)
        y_values = np.arange(-radius_um, radius_um + 0.5 * spacing_um, spacing_um)
        z_values = np.arange(0.0, radius_um + 0.5 * spacing_um, spacing_um)

        baseline_offset_um = self._get_baseline_offset_um(
            center_um,
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis="z",
        )
        grid_offsets_um = self.build_grid_offsets(radius_um=radius_um, spacing_um=spacing_um)
        placement_offsets_um = baseline_offset_um + grid_offsets_um

        results = self._run_offsets(
            placement_offsets_um,
            output_shape=(len(x_values), len(y_values), len(z_values)),
            terminal_um=terminal_um,
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis="z",
            dt=dt,
            v_init=v_init,
        )

        #  results = {
        #     "center_um": center_um,
        #     "center_cell_um": center_cell_um,
        #     "radius_um": float(radius_um),
        #     "spacing_um": float(spacing_um),
        #     "soma_centroid_um": soma_centroid_um,
        #     "baseline_offset_um": baseline_offset_um,
        #     "grid_offsets_um": grid_offsets_um,
        #     "placement_offsets_um": placement_offsets_um,
        #     "terminal_um": float(terminal_um),
        #     "terminal_segment_indices": terminal_indices,
        #     "t_ms": run_results[0]["t_ms"].copy() if run_results else np.array([]),
        #     "terminal_v_mV": np.array(traces),
        #     "terminal_peak_mV": np.array(peaks),
        #     "terminal_segment_v_mV": np.array([run["v_mV"] for run in run_results]),
        #     "runs": run_results,
        # }

        return results

    def run_centered_vertical_sheet(self, center_um=(0.0, 0.0, 0.0), y_radius_um=0.0, z_radius_um=0.0,
                                    y_spacing_um=10.0, z_spacing_um=10.0,
                                    terminal_um=10.0, placement_by="soma", dendrite_um=10.0,
                                    dt=None, v_init=None):
        center_um = np.asarray(center_um, dtype=float)
        if center_um.shape != (3,):
            raise ValueError("center_um must be length 3")

        y_radius_um = float(y_radius_um)
        z_radius_um = float(z_radius_um)
        y_spacing_um = float(y_spacing_um)
        z_spacing_um = float(z_spacing_um)

        x_values = np.array([0.0])
        y_values = np.arange(-y_radius_um, y_radius_um + 0.5 * y_spacing_um, y_spacing_um)
        z_values = np.arange(0.0, z_radius_um + 0.5 * z_spacing_um, z_spacing_um)

        baseline_offset_um = self._get_baseline_offset_um(
            center_um,
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis="z",
        )
        sheet_offsets_um = self.build_vertical_sheet_offsets(
            y_radius_um=y_radius_um,
            z_radius_um=z_radius_um,
            y_spacing_um=y_spacing_um,
            z_spacing_um=z_spacing_um,
        )
        placement_offsets_um = baseline_offset_um + sheet_offsets_um

        return self._run_offsets(
            placement_offsets_um,
            output_shape=(len(x_values), len(y_values), len(z_values)),
            terminal_um=terminal_um,
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis="z",
            dt=dt,
            v_init=v_init,
        )