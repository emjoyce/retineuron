import os
import numpy as np
import matplotlib.pyplot as plt
import skeleton_plot as skpl
from pathlib import Path


def _make_voltage_norm(values, colorbar_mode='auto', fixed_range=None, min_span=1.0):
    """Create a matplotlib Normalize for voltage values with safe padding.

    values may be any array-like; when auto-scaling and the range is tiny or
    degenerate, add a small padding so colors are visible.
    """
    vals = np.asarray(values)
    if colorbar_mode == 'fixed':
        if fixed_range is None:
            raise ValueError('fixed_range must be provided when colorbar_mode="fixed"')
        vmin, vmax = fixed_range
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError('Cannot compute Normalize from non-finite values')
        if np.isclose(vmin, vmax):
            center = vmin
            padding = max(abs(center) * 0.05, min_span / 2)
            vmin = center - padding
            vmax = center + padding
        else:
            span = vmax - vmin
            pad = max(span * 0.05, min_span * 0.05)
            vmin -= pad
            vmax += pad

    return plt.Normalize(vmin=vmin, vmax=vmax)


def plot_electric_potential(electrode, time, axis='z', index=None, xlim=None,
                            ylim=None, figsize=(10, 8), ax=None,
                            cmap='viridis', norm=None, color_bar_vrange=None,
                            alpha=1.0, add_colorbar=True, return_data=False):
    """
    Plot the electric potential at a given time for a specified axis/plane.
    
    Parameters
    ----------
    electrode : Electrode
        Electrode object loaded from electric.py containing voltage data V(x,y,z,t)
    time : float
        Time point at which to plot the potential (in ms)
    axis : str, optional
        Which plane to plot. Options:
        - 'z': Plot x-y plane (default), requires index to specify z position
        - 'y': Plot x-z plane, requires index to specify y position  
        - 'x': Plot y-z plane, requires index to specify x position
    index : int or float, optional
        For 'z' axis: index or value in z array to slice at
        For 'y' axis: index or value in y array to slice at
        For 'x' axis: index or value in x array to slice at
        If None, uses the middle index of the specified axis
    xlim : tuple, optional
        X-axis limits (min, max). If None, defaults to full range
    ylim : tuple, optional
        Y-axis limits (min, max). If None, defaults to full range
        If both xlim and ylim are None, defaults to square plot based on data range
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 8)
    ax : matplotlib axes, optional
        Axes object to plot on. If None, creates a new figure and axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    
    Examples
    --------
    >>> # Plot x-y plane at middle z, at time 10ms
    >>> fig, ax = plot_electric_potential(electrode, time=10.0, axis='z')
    
    >>> # Plot x-z plane at specific y index
    >>> fig, ax = plot_electric_potential(electrode, time=10.0, axis='y', index=50)
    
    >>> # Plot y-z plane at specific x coordinate
    >>> fig, ax = plot_electric_potential(electrode, time=10.0, axis='x', index=100.0)
    
    >>> # Plot with custom limits
    >>> fig, ax = plot_electric_potential(electrode, time=10.0, axis='z', xlim=(-50, 50), ylim=(-50, 50))
    
    >>> # Plot on an existing axes
    >>> fig, ax_custom = plt.subplots(1, 2)
    >>> fig, ax = plot_electric_potential(electrode, time=10.0, axis='z', ax=ax_custom[0])
    """
    
    # Find the closest time index
    time_idx = np.argmin(np.abs(electrode.t - time))
    actual_time = electrode.t[time_idx]
    
    # Set default index to middle if not specified
    if index is None:
        if axis == 'z':
            index = len(electrode.z) // 2
        elif axis == 'y':
            index = len(electrode.y) // 2
        elif axis == 'x':
            index = len(electrode.x) // 2
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Convert float coordinates to nearest idx if needed
    if not isinstance(index, (int, np.integer)):
        if axis == 'z':
            index = np.argmin(np.abs(electrode.z - index))
        elif axis == 'y':
            index = np.argmin(np.abs(electrode.y - index))
        elif axis == 'x':
            index = np.argmin(np.abs(electrode.x - index))
    
    # Extract the 2D slice based on axis
    if axis == 'z':
        # Plot x-y plane at specified z
        voltage_slice = electrode.V[:, :, index, time_idx]
        xlabel = 'x (μm)'
        ylabel = 'y (μm)'
        x_coords = electrode.x
        y_coords = electrode.y
        slice_label = f'z = {electrode.z[index]:.2f} μm'
        
    elif axis == 'y':
        # Plot x-z plane at specified y
        voltage_slice = electrode.V[:, index, :, time_idx]
        xlabel = 'x (μm)'
        ylabel = 'z (μm)'
        x_coords = electrode.x
        y_coords = electrode.z
        slice_label = f'y = {electrode.y[index]:.2f} μm'
        
    elif axis == 'x':
        # Plot y-z plane at specified x
        voltage_slice = electrode.V[index, :, :, time_idx]
        xlabel = 'y (μm)'
        ylabel = 'z (μm)'
        x_coords = electrode.y
        y_coords = electrode.z
        slice_label = f'x = {electrode.x[index]:.2f} μm'
        
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Create the plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
    
    # Plot the voltage as a heatmap
    if norm is None:
        if color_bar_vrange is not None:
            try:
                vmin_val, vmax_val = color_bar_vrange
            except Exception:
                raise ValueError('color_bar_vrange must be (vmin, vmax) or None')
            norm = plt.Normalize(vmin=float(vmin_val), vmax=float(vmax_val))
        else:
            norm = plt.Normalize(vmin=np.nanmin(voltage_slice), vmax=np.nanmax(voltage_slice))

    im = ax.pcolormesh(
        x_coords,
        y_coords,
        voltage_slice.T,
        shading='auto',
        cmap=cmap,
        norm=norm,
        alpha=alpha,
    )
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Electric Potential\n{slice_label}, t = {actual_time:.2f} ms', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar with auto-scaling
    if add_colorbar:
        plt.colorbar(im, ax=ax, label='Potential (mV)', fraction=0.046, pad=0.04)
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    # Set plot limits - default to square plot
    if xlim is None and ylim is None:
        # Default to square plot based on data range
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        half_size = max((x_max - x_min) / 2, (y_max - y_min) / 2)
        xlim = (x_center - half_size, x_center + half_size)
        ylim = (y_center - half_size, y_center + half_size)
    else:
        # Use provided limits or full range
        if xlim is None:
            xlim = (x_coords.min(), x_coords.max())
        if ylim is None:
            ylim = (y_coords.min(), y_coords.max())
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if created_fig:
        fig.tight_layout()

    if return_data:
        data = {
            'time_idx': time_idx,
            'actual_time': actual_time,
            'voltage_slice': voltage_slice,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'slice_label': slice_label,
            'norm': norm,
            'mappable': im,
        }
        return fig, ax, data

    return fig, ax



def plot_cell_extracellular_voltage(cell, simulation, time, x='x', y='y', 
                                    cmap='viridis', line_width=2, plot_soma=False,
                                    soma_size=300, ax=None, norm=None,
                                    color_bar_vrange=None,
                                    add_colorbar=True, return_data=False):
    """
    Plot cell morphology colored by extracellular voltage at a given time.
    
    Uses skeleton_plot.plot_tools.plot_verts with skel_colors set to the 
    extracellular voltage at each vertex.
    
    Parameters
    ----------
    cell : Cell
        Cell object with verts, edges, and radii already loaded.
    simulation : ExtracellularSimulation
        Simulation object with computed extracellular potentials at segments.
    time : float
        Time point to plot, in ms.
    x : str, optional
        Which axis for x. Options: 'x', 'y', 'z'. Default is 'x'.
    y : str, optional
        Which axis for y. Options: 'x', 'y', 'z'. Default is 'y'.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    line_width : float, optional
        Line width for skeleton. Default is 2.
    plot_soma : bool, optional
        Whether to plot soma. Default is False.
    soma_size : float, optional
        Soma marker size. Default is 300.
    ax : matplotlib axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    figsize : tuple, optional
        Figure size if creating a new figure. Default is (10, 8).
    color_bar_vrange : tuple(float, float) or None, optional
        Optional fixed color limits for the colorbar as `(vmin, vmax)`. If
        provided they override automatic scaling (used only when `norm` is
        not supplied).
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> fig, ax = plot_cell_extracellular_voltage(cell, sim, time=10.0, x='x', y='y')
    """
    
    # Compute potentials if not already done
    if getattr(simulation, 'v_external_seg', None) is None:
        simulation.compute_segment_potentials()
    
    # Find closest time index
    t_values = np.asarray(simulation.t_field)
    time_idx = np.argmin(np.abs(t_values - time))
    actual_time = t_values[time_idx]
    
    # Get data
    verts = np.asarray(cell.verts)
    edges = np.asarray(cell.edges, dtype=int)
    seg_xyz = np.asarray(cell.seg_xyz)
    ve_seg = np.asarray(simulation.v_external_seg)[:, time_idx]
    
    # Map segment voltage to each vertex by finding nearest segment
    n_verts = verts.shape[0]
    ve_verts = np.zeros(n_verts)
    
    for v_idx in range(n_verts):
        # Find nearest segment center
        distances = np.linalg.norm(verts[v_idx] - seg_xyz, axis=1)
        nearest_seg = np.argmin(distances)
        ve_verts[v_idx] = ve_seg[nearest_seg]
    
    # Create figure and plot using skeleton_plot
    created_fig = False
    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()
    else:
        fig = ax.get_figure()
    
    # Create colormap and normalize voltages
    from matplotlib.cm import get_cmap
    if norm is None:
        if color_bar_vrange is not None:
            try:
                vmin_val, vmax_val = color_bar_vrange
            except Exception:
                raise ValueError('color_bar_vrange must be (vmin, vmax) or None')
            norm = plt.Normalize(vmin=float(vmin_val), vmax=float(vmax_val))
        else:
            norm = plt.Normalize(vmin=np.nanmin(ve_verts), vmax=np.nanmax(ve_verts))
    color_map = get_cmap(cmap)
    
    # Map voltage values to colors for each vertex index
    ve_verts_normalized = norm(ve_verts)
    skel_color_map = {i: color_map(ve_verts_normalized[i]) for i in range(n_verts)}
    
    # skel_colors should contain indices that map to skel_color_map keys
    skel_colors = np.arange(n_verts)
    
    # Plot using skeleton_plot
    skpl.plot_tools.plot_verts(verts, edges, 
                               radius=cell.radii,
                               x=x, y=y,
                               skel_colors=skel_colors,
                               skel_color_map=skel_color_map,
                               plot_soma=plot_soma,
                               soma_size=soma_size,
                               line_width=line_width,
                               ax=ax)
    
    ax.set_title(f'Cell extracellular voltage at t = {actual_time:.2f} ms')
    
    # Add colorbar
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(
            sm,
            ax=ax,
            label='Extracellular potential (mV)',
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
        )

    if return_data:
        data = {
            'time_idx': time_idx,
            'actual_time': actual_time,
            've_verts': ve_verts,
            'norm': norm,
        }
        return fig, ax, data

    return fig, ax


def plot_cell_transmembrane_voltage(cell, simulation, time, x='x', y='y', 
                                    cmap='viridis', line_width=2, plot_soma=False,
                                    soma_size=300, ax=None, xlim=None, ylim=None,
                                    figsize=(10, 8), norm=None, color_bar_vrange=None,
                                    add_colorbar=True, return_data=False):
    """
    Plot cell morphology colored by intracellular voltage at a given time.

    This uses recorded NEURON membrane voltage from simulation.results.
    
    Parameters
    ----------
    cell : Cell
        Cell object with verts, edges, and radii already loaded.
    simulation : ExtracellularSimulation
        Simulation object after run() has been called.
    time : float
        Time point to plot, in ms.
    x : str, optional
        Which axis for x. Options: 'x', 'y', 'z'. Default is 'x'.
    y : str, optional
        Which axis for y. Options: 'x', 'y', 'z'. Default is 'y'.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    line_width : float, optional
        Line width for skeleton. Default is 2.
    plot_soma : bool, optional
        Whether to plot soma. Default is False.
    soma_size : float, optional
        Soma marker size. Default is 300.
    ax : matplotlib axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    xlim : tuple, optional
        X-axis limits. If None, uses full range.
    ylim : tuple, optional
        Y-axis limits. If None, uses full range.
    figsize : tuple, optional
        Figure size if creating a new figure. Default is (10, 8).
    norm : matplotlib.colors.Normalize, optional
        Shared normalization for coloring.
    add_colorbar : bool, optional
        Whether to add a colorbar. Default is True.
    return_data : bool, optional
        If True, return the computed vertex voltages and normalization.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """

    if not hasattr(simulation, 'results') or simulation.results is None:
        raise ValueError('simulation.results is missing; run simulation.run() first')

    results = simulation.results
    t_values = np.asarray(results['t_ms'])
    time_idx = np.argmin(np.abs(t_values - time))
    actual_time = t_values[time_idx]

    v_mV = np.asarray(results['v_mV'])
    recorded_seg_indices = np.asarray(results['recorded_seg_indices'])
    if v_mV.ndim != 2:
        raise ValueError(f"simulation.results['v_mV'] should be 2D, got shape {v_mV.shape}")

    if v_mV.shape[0] != len(recorded_seg_indices):
        raise ValueError(
            f"v_mV has {v_mV.shape[0]} traces but recorded_seg_indices has {len(recorded_seg_indices)} entries"
        )

    verts = np.asarray(cell.verts)
    seg_xyz = np.asarray(cell.seg_xyz)
    cell_seg_xyz = seg_xyz[recorded_seg_indices] if len(recorded_seg_indices) else seg_xyz
    v_seg = v_mV[:, time_idx]

    n_verts = verts.shape[0]
    v_verts = np.zeros(n_verts)
    for v_idx in range(n_verts):
        distances = np.linalg.norm(verts[v_idx] - cell_seg_xyz, axis=1)
        nearest_seg = np.argmin(distances)
        v_verts[v_idx] = v_seg[nearest_seg]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    from matplotlib.cm import get_cmap
    if norm is None:
        if color_bar_vrange is not None:
            try:
                vmin_val, vmax_val = color_bar_vrange
            except Exception:
                raise ValueError('color_bar_vrange must be (vmin, vmax) or None')
            norm = plt.Normalize(vmin=float(vmin_val), vmax=float(vmax_val))
        else:
            norm = plt.Normalize(vmin=np.nanmin(v_verts), vmax=np.nanmax(v_verts))
    color_map = get_cmap(cmap)
    v_verts_normalized = norm(v_verts)
    skel_color_map = {i: color_map(v_verts_normalized[i]) for i in range(n_verts)}
    skel_colors = np.arange(n_verts)

    skpl.plot_tools.plot_verts(
        verts,
        np.asarray(cell.edges, dtype=int),
        radius=cell.radii,
        x=x,
        y=y,
        skel_colors=skel_colors,
        skel_color_map=skel_color_map,
        plot_soma=plot_soma,
        soma_size=soma_size,
        line_width=line_width,
        ax=ax,
    )

    ax.set_title(f'Cell intracellular voltage at t = {actual_time:.2f} ms')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(
            sm,
            ax=ax,
            label='Membrane potential (mV)',
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
        )

    if created_fig:
        fig.tight_layout()

    if return_data:
        data = {
            'time_idx': time_idx,
            'actual_time': actual_time,
            'v_verts': v_verts,
            'norm': norm,
        }
        return fig, ax, data

    return fig, ax


def plot_cell_on_electric(cell, simulation, electrode, time, axis='z', index=None,
                          xlim=None, ylim=None, figsize=(10, 8), ax=None,
                          x=None, y=None, cmap='viridis', norm=None,
                          cell_norm=None,
                          color_bar_vrange=None,
                          alpha=0.9, line_width=2, plot_soma=False,
                          soma_size=300, add_colorbar=True,
                          cell_voltage='extracellular', cell_time=None,
                          return_data=False):
    """
    Plot neuron morphology on top of electric potential.

    time is NEURON-relative time.
    field_time is RPSim/electrode absolute time.
    """

    if cell_time is None:
        cell_time = time

    # Convert NEURON-relative time to RPSim/electrode absolute time.
    field_time = time + electrode.t[0]

    # First draw temporary field to get field norm/data, then remove it.
    fig, ax, field_data = plot_electric_potential(
        electrode,
        field_time,
        axis=axis,
        index=index,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        ax=ax,
        cmap=cmap,
        norm=norm,
        color_bar_vrange=color_bar_vrange,
        alpha=alpha,
        add_colorbar=False,
        return_data=True,
    )

    field_data['mappable'].remove()
    field_norm = field_data['norm']

    # Your existing skeleton_plot axis convention.
    if x is None or y is None:
        if axis == 'z':
            skel_x, skel_y = 'x', 'z'
        elif axis == 'y':
            skel_x, skel_y = 'x', 'y'
        else:
            skel_x, skel_y = 'z', 'y'
    else:
        skel_x, skel_y = x, y

    # Draw neuron first.
    if cell_voltage == 'extracellular':
        # Extracellular neuron and field use the same color scale.
        _, _, cell_data = plot_cell_extracellular_voltage(
            cell,
            simulation,
            field_time,
            x=skel_x,
            y=skel_y,
            cmap=cmap,
            line_width=line_width,
            plot_soma=plot_soma,
            soma_size=soma_size,
            ax=ax,
            norm=field_norm,
            color_bar_vrange=color_bar_vrange,
            add_colorbar=False,
            return_data=True,
        )

    elif cell_voltage == 'intracellular':
        # Intracellular Vm needs its own color scale.
        _, _, cell_data = plot_cell_transmembrane_voltage(
            cell,
            simulation,
            cell_time,
            x=skel_x,
            y=skel_y,
            cmap=cmap,
            line_width=line_width,
            plot_soma=plot_soma,
            soma_size=soma_size,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            norm=cell_norm,
            color_bar_vrange=color_bar_vrange,
            add_colorbar=False,
            return_data=True,
        )

    else:
        raise ValueError("cell_voltage must be 'extracellular' or 'intracellular'")

    # Draw field second.
    _, _, field_data = plot_electric_potential(
        electrode,
        field_time,
        axis=axis,
        index=index,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        ax=ax,
        cmap=cmap,
        norm=field_norm,
        color_bar_vrange=color_bar_vrange,
        alpha=alpha,
        add_colorbar=False,
        return_data=True,
    )

    if add_colorbar:
        if cell_voltage == 'extracellular':
            # One shared colorbar.
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=field_norm)
            sm.set_array([])
            plt.colorbar(
                sm,
                ax=ax,
                label='Extracellular potential (mV)',
                fraction=0.046,
                pad=0.04,
            )

        else:
            # Two colorbars: field and intracellular Vm.
            # Place them in separate axes so they do not overlap.
            fig = ax.get_figure()
            bbox = ax.get_position()
            bar_width = 0.022
            plot_gap = -.1
            bar_gap = 0.05

            # Reserve enough room on the right for both bars.
            fig.subplots_adjust(right=min(bbox.x1, 0.72))

            cax_field = fig.add_axes([
                bbox.x1 + plot_gap,
                bbox.y0,
                bar_width,
                bbox.height,
            ])
            cax_cell = fig.add_axes([
                bbox.x1 + plot_gap + bar_width + bar_gap,
                bbox.y0,
                bar_width,
                bbox.height,
            ])

            sm_field = plt.cm.ScalarMappable(cmap=cmap, norm=field_norm)
            sm_field.set_array([])
            plt.colorbar(
                sm_field,
                cax=cax_field,
                label='Field potential (mV)',
            )
            cax_field.yaxis.set_label_position('left')
            cax_field.yaxis.tick_left()
            cax_field.yaxis.labelpad = 1

            sm_cell = plt.cm.ScalarMappable(cmap=cmap, norm=cell_data['norm'])
            sm_cell.set_array([])
            plt.colorbar(
                sm_cell,
                cax=cax_cell,
                label='Membrane voltage (mV)',
            )

    ax.set_title(
        f"Cell on Electric Field\n{field_data['slice_label']}, "
        f"cell t = {cell_time:.2f} ms"
    )

    if return_data:
        data = {
            'field': field_data,
            'cell': cell_data,
            'field_norm': field_norm,
            'cell_norm': cell_data['norm'],
        }
        return fig, ax, data

    return fig, ax


def save_voltage_sequence(cell, simulation, electrode, out_dir,
                          n_images=40,
                          times=None,
                          time_start=None,
                          time_stop=None,
                          cell_voltage='intracellular',
                          axis='x',
                          index=None,
                          xlim=None,
                          ylim=None,
                          figsize=(10, 8),
                          ax_kwargs=None,
                          x=None,
                          y=None,
                          cmap='viridis',
                          alpha=0.9,
                          line_width=2,
                          plot_soma=False,
                          soma_size=300,
                          dpi=150,
                          filename_prefix='frame'):
    """
    Save a sequence of frames by repeatedly calling plot_cell_on_electric().

    This intentionally does not reinterpret time. The times passed here are
    passed directly to plot_cell_on_electric(), exactly like your manual call.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose times to plot.
    if times is None:
        if time_start is None:
            time_start = simulation.results['t_ms'][0]
        if time_stop is None:
            time_stop = simulation.results['t_ms'][-1]

        times = np.linspace(time_start, time_stop, n_images)
    else:
        times = np.asarray(times)
        n_images = len(times)

    # Make fixed norms across all frames.
    if cell_voltage == 'extracellular':
        if getattr(simulation, 'v_external_seg', None) is None:
            simulation.compute_segment_potentials()

        all_vals = np.concatenate([
            np.asarray(electrode.V).ravel(),
            np.asarray(simulation.v_external_seg).ravel(),
        ])

        field_norm = _make_voltage_norm(all_vals)
        cell_norm = field_norm

    elif cell_voltage == 'intracellular':
        field_norm = _make_voltage_norm(np.asarray(electrode.V))
        cell_norm = _make_voltage_norm(np.asarray(simulation.results['v_mV']))

    else:
        raise ValueError("cell_voltage must be 'extracellular' or 'intracellular'")

    # Save frames.
    for i, t in enumerate(times):
        fig, ax = plt.subplots(figsize=figsize)

        plot_cell_on_electric(
            cell,
            simulation,
            electrode,
            time=t,
            axis=axis,
            index=index,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            ax=ax,
            x=x,
            y=y,
            cmap=cmap,
            norm=field_norm,
            cell_norm=cell_norm,
            alpha=alpha,
            line_width=line_width,
            plot_soma=plot_soma,
            soma_size=soma_size,
            add_colorbar=True,
            cell_voltage=cell_voltage,
            cell_time=t,
        )

        save_path = out_dir / f"{filename_prefix}_{i:04d}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {n_images} frames to {out_dir}")