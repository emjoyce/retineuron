import numpy as np
import matplotlib.pyplot as plt
from neuron import h

# funcs for stitchign together e field sims 
def _copy_simulation_state(state):
    copied_state = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied_state[key] = value.copy()
        else:
            copied_state[key] = value
    return copied_state

def _stitch_trace_arrays(first_array, second_array):
    first_array = np.asarray(first_array)
    second_array = np.asarray(second_array)

    if first_array.size == 0:
        return second_array.copy()
    if second_array.size == 0:
        return first_array.copy()

    return np.concatenate([first_array, second_array[..., 1:]], axis=-1)

def _stitch_time_vectors(first_t_ms, second_t_ms):
    first_t_ms = np.asarray(first_t_ms, dtype=float)
    second_t_ms = np.asarray(second_t_ms, dtype=float)

    if first_t_ms.size == 0:
        return second_t_ms.copy()
    if second_t_ms.size == 0:
        return first_t_ms.copy()

    shifted_second_t_ms = first_t_ms[-1] + second_t_ms[1:]
    return np.concatenate([first_t_ms, shifted_second_t_ms])

def _stitch_trace_sequence(arrays):
    arrays = [np.asarray(array) for array in arrays]
    if not arrays:
        return np.array([])

    stitched = arrays[0].copy()
    for array in arrays[1:]:
        stitched = _stitch_trace_arrays(stitched, array)

    return stitched

def _stitch_time_sequence(time_vectors):
    time_vectors = [np.asarray(time_vector, dtype=float) for time_vector in time_vectors]
    if not time_vectors:
        return np.array([], dtype=float)

    stitched = time_vectors[0].copy()
    for time_vector in time_vectors[1:]:
        stitched = _stitch_time_vectors(stitched, time_vector)

    return stitched

class ExtracellularSimulation:
    def __init__(self, cell, electrode):
        self.cell = cell
        self.electrode = electrode

        self.t_field = np.array(electrode.t)
        self.t_play = self.t_field - self.t_field[0]

        self.v_external_seg = None
        self.ve_vectors = []
        self.initial_state = None
        self.last_state = None

    def get_field_dt_ms(self):
        if self.t_play.size >= 2:
            dt_values = np.diff(self.t_play)
            dt_ms = float(dt_values[0])
            return dt_ms

        full_t_play = np.asarray(self.electrode.t, dtype=float)
        if full_t_play.size < 2:
            raise ValueError("Electrode time grid must contain at least two points to infer dt")

        full_t_play = full_t_play - full_t_play[0]
        dt_values = np.diff(full_t_play)
        dt_ms = float(dt_values[0])
        return dt_ms

    def _normalize_stop_time_idx(self, stop_time_idx):
        if stop_time_idx is None:
            return None

        stop_time_idx = int(stop_time_idx)
        n_t = len(self.electrode.t)
        if not (0 <= stop_time_idx < n_t):
            raise IndexError(f"stop_time_idx must be between 0 and {n_t - 1}")

        return stop_time_idx

    def _get_state_dict(self, state):

        segment_v_mV = np.asarray(state["segment_v_mV"], dtype=float)
        if segment_v_mV.ndim != 1:
            raise ValueError("state['segment_v_mV'] must be one-dimensional")
        if segment_v_mV.shape[0] != len(self.cell.seg_refs):
            raise ValueError("state segment count does not match the cell segment count")

        return {
            "segment_v_mV": segment_v_mV.copy(),
            "t_ms": float(state.get("t_ms", 0.0)),
            "n_segments": int(segment_v_mV.shape[0]),
        }

    def get_last_state(self):
        state = {
            "segment_v_mV": np.array([seg.v for seg in self.cell.seg_refs], dtype=float),
            "t_ms": float(h.t),
            "n_segments": len(self.cell.seg_refs),
        }
        self.last_state = _copy_simulation_state(state)
        return _copy_simulation_state(state)

    def load_state(self, state):
        self.initial_state = self._get_state_dict(state)
        return _copy_simulation_state(self.initial_state)

    def _restore_loaded_state(self):
        if self.initial_state is None:
            return

        for seg_idx, seg in enumerate(self.cell.seg_refs):
            seg.v = float(self.initial_state["segment_v_mV"][seg_idx])

        h.fcurrent()
        h.frecord_init()

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
    
    def apply_extracellular_potential(self, position_offset_um=None, stop_time_idx=None):
        """
        apply the electrode voltage over time into every NEURON compartment

        Uses:
            self.v_external_seg[segment, time] in mV
            self.electrode.t in ms
        """

        # compute volt at each seg, only if not alredy done
        if self.v_external_seg is None or self.v_external_seg.shape[1] != len(self.electrode.t):
            self.compute_segment_potentials(position_offset_um=position_offset_um)

        stop_time_idx = self._normalize_stop_time_idx(stop_time_idx)

        # shifting time to start at 0
        self.t_field = np.array(self.electrode.t)
        if stop_time_idx is not None:
            stop_slice = slice(None, stop_time_idx + 1)
            self.t_field = self.t_field[stop_slice]
            self.v_external_seg = self.v_external_seg[:, stop_slice]
        self.t_play = self.t_field - self.t_field[0]

        # save time vector
        self.t_vec = h.Vector(self.t_play)

        self.ve_vectors = []

        # play one extracellular voltage over time into each segment.
        for seg_idx, seg in enumerate(self.cell.seg_refs):
            ve_vec = h.Vector(self.v_external_seg[seg_idx, :])
            ve_vec.play(seg._ref_e_extracellular, self.t_vec, True)
            self.ve_vectors.append(ve_vec)
    def run(self, dt = None, v_init=None, record_segments=None, position_offset_um=None,
        stop_time_idx=None):
        """
        Apply extracellular stimulation, record response, and run NEURON.

        dt: NEURON integration time step in ms. If None, infer from the
            electrode time grid.
        v_init: initial membrane potential in mV
        record_segments: optional list of segment indices to record
        position_offset_um: optional placement offset applied only when sampling
            the extracellular field
        stop_time_idx: optional inclusive index into the electrode time bins.
            If provided, only run through that field sample.
        """

        # Use the cell's passive reversal as default initialization voltage.
        if v_init is None:
            v_init = self.cell.e_pas

        # Put the RPSim extracellular potential into NEURON.
        self.apply_extracellular_potential(
            position_offset_um=position_offset_um,
            stop_time_idx=stop_time_idx,
        )

        # Set up recordings before running.
        self.record(seg_indices=record_segments)

        # Run the simulation.
        if dt is None:
            dt = self.get_field_dt_ms()

        h.dt = dt
        h.tstop = float(self.t_play[-1])

        h.finitialize(v_init)
        self._restore_loaded_state()
        h.continuerun(h.tstop)

        final_state = self.get_last_state()

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
            "initial_state": None if self.initial_state is None else _copy_simulation_state(self.initial_state),
            "final_state": final_state,
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

        # option to just record some segments, i.e. just terminals
        if seg_indices is None:
            seg_indices = range(len(self.cell.seg_refs))

        self.recorded_seg_indices = list(seg_indices)

        # keep time vec
        self.t_rec = h.Vector().record(h._ref_t)

        self.v_rec = []
        self.vext_rec = []
        self.eext_rec = []

        for seg_idx in self.recorded_seg_indices:
            seg = self.cell.seg_refs[seg_idx]

            self.v_rec.append(h.Vector().record(seg._ref_v))
            self.vext_rec.append(h.Vector().record(seg._ref_vext[0]))
            self.eext_rec.append(h.Vector().record(seg._ref_e_extracellular))

# class that stitches two extracellular simulations togehter. 
# needed if we want the object returned that can be used for plotting and analysis consistency and smoothness
class ExtracellularStitchedStimulation:
    def __init__(self, cell, electrodes):
        self.cell = cell

        try:
            self.electrodes = list(electrodes)
        except TypeError as exc:
            raise TypeError("electrodes must be an iterable of electrode objects") from exc

        if len(self.electrodes) == 0:
            raise ValueError("electrodes must contain at least one electrode")

        self.first_electrode = self.electrodes[0]
        self.second_electrode = self.electrodes[1] if len(self.electrodes) > 1 else None

        self.phase_sims = []
        self.phase_results = []
        self.first_sim = None
        self.second_sim = None

        self.initial_state = None
        self.transition_states = []
        self.transition_state = None
        self.last_state = None

        self.v_external_seg = None
        self.t_field = np.array([], dtype=float)
        self.t_play = np.array([], dtype=float)
        self.field_dt_ms = np.array([], dtype=float)

        self.sampled_seg_xyz_um = None
        self.sampled_seg_xyz_electrode_um = None
        self.results = None

    def _sync_phase_aliases(self):
        self.first_sim = self.phase_sims[0] if len(self.phase_sims) >= 1 else None
        self.second_sim = self.phase_sims[1] if len(self.phase_sims) >= 2 else None
        self.transition_state = (
            _copy_simulation_state(self.transition_states[0])
            if len(self.transition_states) == 1
            else None
        )

    def _refresh_stitched_field_data(self):
        if not self.phase_sims:
            raise ValueError("At least one phase simulation must exist before stitching field data")
        if any(phase_sim.v_external_seg is None for phase_sim in self.phase_sims):
            raise ValueError("All phase simulations must have sampled field data before stitching")

        self.v_external_seg = _stitch_trace_sequence(
            [phase_sim.v_external_seg for phase_sim in self.phase_sims]
        )
        self.t_field = _stitch_time_sequence([phase_sim.t_play for phase_sim in self.phase_sims])
        self.t_play = self.t_field.copy()
        last_phase_sim = self.phase_sims[-1]
        self.sampled_seg_xyz_um = last_phase_sim.sampled_seg_xyz_um.copy()
        self.sampled_seg_xyz_electrode_um = last_phase_sim.sampled_seg_xyz_electrode_um.copy()

    def get_field_dt_ms(self):
        self.field_dt_ms = np.array([
            ExtracellularSimulation(self.cell, electrode).get_field_dt_ms()
            for electrode in self.electrodes
        ], dtype=float)
        return self.field_dt_ms.copy()

    def load_state(self, state):
        validator = ExtracellularSimulation(self.cell, self.first_electrode)
        self.initial_state = validator._get_state_dict(state)
        return _copy_simulation_state(self.initial_state)

    def get_last_state(self):
        if self.last_state is None:
            raise ValueError("No final state is available; run the stitched simulation first")

        return _copy_simulation_state(self.last_state)

    def _get_dt_sequence(self, dt_sequence):
        if dt_sequence is None:
            return [None] * len(self.electrodes)

        if np.isscalar(dt_sequence):
            dt_value = float(dt_sequence)
            return [dt_value] * len(self.electrodes)

        dt_sequence = list(dt_sequence)
        if len(dt_sequence) != len(self.electrodes):
            raise ValueError("dt_sequence must match the number of electrodes")

        normalized_dt_sequence = []
        for dt_value in dt_sequence:
            normalized_dt_sequence.append(None if dt_value is None else float(dt_value))

        return normalized_dt_sequence

    def _get_stop_phase_indices(self, stop_electrode):
        if stop_electrode is None:
            return []

        if isinstance(stop_electrode, (int, np.integer)):
            stop_idx = int(stop_electrode)
            if not (0 <= stop_idx < len(self.electrodes)):
                raise IndexError(f"stop_electrode must be between 0 and {len(self.electrodes) - 1}")
            return [stop_idx]

        stop_indices = [
            electrode_idx
            for electrode_idx, electrode in enumerate(self.electrodes)
            if electrode is stop_electrode
        ]
        if not stop_indices:
            raise ValueError("stop_electrode must be one of the electrodes in this stitched simulation")

        return stop_indices

    def compute_segment_potentials(self, position_offset_um=None):
        self.phase_sims = [
            ExtracellularSimulation(self.cell, electrode)
            for electrode in self.electrodes
        ]

        for phase_sim in self.phase_sims:
            phase_sim.compute_segment_potentials(position_offset_um=position_offset_um)

        self._sync_phase_aliases()
        self._refresh_stitched_field_data()
        return self.v_external_seg

    def run(self, v_init=None, record_segments=None, position_offset_um=None,
            dt_sequence=None, stop_electrode=None, stop_time_idx=None):
        dt_sequence = self._get_dt_sequence(dt_sequence)

        if (stop_electrode is None) != (stop_time_idx is None):
            raise ValueError("stop_electrode and stop_time_idx must both be provided or both be None")

        stop_phase_indices = self._get_stop_phase_indices(stop_electrode)
        stop_phase_index_set = set(stop_phase_indices)

        self.phase_sims = []
        self.phase_results = []
        self.transition_states = []

        current_state = None if self.initial_state is None else _copy_simulation_state(self.initial_state)

        for phase_idx, (electrode, dt_value) in enumerate(zip(self.electrodes, dt_sequence)):
            phase_sim = ExtracellularSimulation(self.cell, electrode)
            if current_state is not None:
                phase_sim.load_state(current_state)

            phase_stop_time_idx = stop_time_idx if phase_idx in stop_phase_index_set else None

            phase_result = phase_sim.run(
                dt=dt_value,
                v_init=v_init,
                record_segments=record_segments,
                position_offset_um=position_offset_um,
                stop_time_idx=phase_stop_time_idx,
            )

            self.phase_sims.append(phase_sim)
            self.phase_results.append(phase_result)

            current_state = phase_sim.get_last_state()
            if phase_idx < len(self.electrodes) - 1:
                self.transition_states.append(_copy_simulation_state(current_state))

        self._sync_phase_aliases()
        self._refresh_stitched_field_data()
        self.field_dt_ms = np.array([
            phase_result["dt_ms"] for phase_result in self.phase_results
        ], dtype=float)
        self.last_state = _copy_simulation_state(self.phase_results[-1]["final_state"])

        stitched_t_ms = _stitch_time_sequence([
            phase_result["t_ms"] for phase_result in self.phase_results
        ])
        self.t_field = stitched_t_ms.copy()
        self.t_play = stitched_t_ms.copy()

        last_phase_result = self.phase_results[-1]

        self.results = {
            "t_ms": stitched_t_ms,
            "v_mV": _stitch_trace_sequence([
                phase_result["v_mV"] for phase_result in self.phase_results
            ]),
            "vext_mV": _stitch_trace_sequence([
                phase_result["vext_mV"] for phase_result in self.phase_results
            ]),
            "eext_mV": _stitch_trace_sequence([
                phase_result["eext_mV"] for phase_result in self.phase_results
            ]),
            "Ve_input_mV": self.v_external_seg.copy(),
            "recorded_seg_indices": self.phase_results[0]["recorded_seg_indices"].copy(),
            "seg_xyz_um": self.phase_results[0]["seg_xyz_um"].copy(),
            "sampled_seg_xyz_um": self.sampled_seg_xyz_um.copy(),
            "sampled_seg_xyz_electrode_um": self.sampled_seg_xyz_electrode_um.copy(),
            "position_offset_um": last_phase_result["position_offset_um"].copy(),
            "initial_state": None if self.initial_state is None else _copy_simulation_state(self.initial_state),
            "final_state": _copy_simulation_state(self.last_state),
            "field_dt_ms": self.field_dt_ms.copy(),
            "stop_electrode_indices": np.array(stop_phase_indices, dtype=int),
            "stop_time_idx": None if stop_time_idx is None else int(stop_time_idx),
            "transition_states": [
                _copy_simulation_state(state) for state in self.transition_states
            ],
            "phases": list(self.phase_results),
        }

        if len(self.phase_results) == 2:
            self.results["first"] = self.phase_results[0]
            self.results["second"] = self.phase_results[1]
            self.results["transition_state"] = _copy_simulation_state(self.transition_states[0])

        return self.results

# can also use this to record just the terminal verts over a many electrode sim byt setting record segments
def run_field_sequence(cell, electrodes, v_init=None, record_segments=None,
                       position_offset_um=None, dt_sequence=None,
                       stop_electrode=None, stop_time_idx=None):
    """
    Run any number of extracellular fields back-to-back on the same cell.

    'electrodes' must be an iterable of electrode objects in execution order.
    If 'dt_sequence' is omitted, each phase pulls 'dt' from its own field.
    """

    stitched_sim = ExtracellularStitchedStimulation(cell, electrodes)
    stitched_sim.run(
        v_init=v_init,
        record_segments=record_segments,
        position_offset_um=position_offset_um,
        dt_sequence=dt_sequence,
        stop_electrode=stop_electrode,
        stop_time_idx=stop_time_idx,
    )
    return stitched_sim

# func for placing cell in many locations in field and geting terminal volt
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
                          dt=None, v_init=None, z_radius_um=None, z_spacing_um=None):
        # run cell sim at multiple placements in the electric field and pull out terminal responses
        center_um = np.asarray(center_um, dtype=float)
        if center_um.shape != (3,):
            raise ValueError("center_um must be length 3")

        radius_um = float(radius_um)
        spacing_um = float(spacing_um)
        z_radius_um = radius_um if z_radius_um is None else float(z_radius_um)
        z_spacing_um = spacing_um if z_spacing_um is None else float(z_spacing_um)
        x_values = np.arange(-radius_um, radius_um + 0.5 * spacing_um, spacing_um)
        y_values = np.arange(-radius_um, radius_um + 0.5 * spacing_um, spacing_um)
        z_values = np.arange(0.0, z_radius_um + 0.5 * z_spacing_um, z_spacing_um)

        baseline_offset_um = self._get_baseline_offset_um(
            center_um,
            placement_by=placement_by,
            dendrite_um=dendrite_um,
            axis="z",
        )
        grid_offsets_um = self.build_grid_offsets(radius_um=radius_um, spacing_um=spacing_um)
        if z_radius_um != radius_um or z_spacing_um != spacing_um:
            grid_3d = np.meshgrid(x_values, y_values, z_values, indexing="ij")
            grid_offsets_um = np.column_stack([axis.ravel() for axis in grid_3d])
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