import numpy as np
import matplotlib.pyplot as plt
import skeleton_plot as skpl


def plot_electric_potential(electrode, time, axis='z', index=None, xlim=None, 
                            ylim=None, figsize=(10, 8), ax=None):
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
    im = ax.pcolormesh(x_coords, y_coords, voltage_slice.T, 
                       shading='auto', cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Electric Potential\n{slice_label}, t = {actual_time:.2f} ms', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar with auto-scaling
    cbar = plt.colorbar(im, ax=ax, label='Potential (mV)', fraction=0.046, pad=0.04)
    
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
    return fig, ax



def plot_cell_extracellular_voltage(cell, simulation, time, x='x', y='y', 
                                    cmap='viridis', line_width=2, plot_soma=False,
                                    soma_size=300, ax=None):
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
    t_values = np.asarray(getattr(simulation, 't_field', simulation.t_play))
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
    norm = plt.Normalize(vmin=np.min(ve_verts), vmax=np.max(ve_verts))
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        label='Extracellular potential (mV)',
        fraction=0.046,
        pad=0.04,
        shrink=0.9,
    )
    
    return fig, ax

