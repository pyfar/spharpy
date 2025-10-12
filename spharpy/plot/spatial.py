"""
Plot functions for spatial data.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.spatial as sspat
import pyfar as pf
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
__all__ = [Axes3D]
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packaging import version
from scipy.stats import circmean

from .cmap import phase_twilight

from spharpy.samplings import spherical_voronoi
from pyfar.classes.coordinates import sph2cart


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def scatter(coordinates, ax=None, **kwargs):
    """Plot the x, y, and z coordinates of the sampling grid in the 3d space.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        The coordinates to be plotted
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object.
    **kwargs : optional
        Additional keyword arguments passed to the scatter function.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> coords = spharpy.samplings.gaussian(n_max=5)
        >>> spharpy.plot.scatter(coords)


    """
    if not isinstance(coordinates, pf.Coordinates):
        raise ValueError("coordinates must be a coordinates object.")

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection='3d')

    if '3d' not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    ax.scatter(coordinates.x, coordinates.y, coordinates.z, **kwargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([
        np.ptp(coordinates.x),
        np.ptp(coordinates.y),
        np.ptp(coordinates.z)])

    return ax


def _triangulation_sphere(sampling, data):
    """Triangulation for data points sampled on a spherical surface.

    Parameters
    ----------
    sampling : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinate object for which the triangulation is calculated
    data : list of arrays
        x, y, and z values of the data points in the triangulation

    Returns
    -------
    triangulation : matplotlib Triangulation

    """

    x, y, z = sph2cart(
        sampling.azimuth,
        sampling.colatitude,
        np.abs(data),
        )
    hull = sspat.ConvexHull(
        np.asarray(sph2cart(
            sampling.azimuth,
            sampling.colatitude,
            np.ones(len(sampling.colatitude)))).T)
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)

    return tri, [x, y, z]


def interpolate_data_on_sphere(
        sampling,
        data,
        overlap=np.pi*0.25,
        refine=False,
        interpolator='linear'):
    """Linear interpolator for data on a spherical surface. The interpolator
    exploits that the data on the sphere is periodic with regard to the
    elevation and azimuth angle. The data is periodically extended to a
    specified overlap angle before interpolation.

    Parameters
    ----------
    sampling : pyfar.Coordinates, spharpy.SamplingSphere
        The coordinates at which the data is sampled.
    data : ndarray, double
        The sampled data points.
    overlap : float, (pi/4)
        The overlap for the periodic extension in azimuth angle, given in
        radians
    refine : bool
        Refine the mesh before interpolating. The default is ``False``.
    interpolator : 'linear', 'cubic'
        The interpolation method to be used. The default is 'linear'.

    Returns
    -------
    interp : LinearTriInterpolator, CubicTriInterpolator
        The interpolator object.

    Note
    ----
    Internally, matplotlibs LinearTriInterpolator or CubicTriInterpolator
    are used.

    """
    _, lats, lons = coordinates2latlon(sampling)

    mask = lons > np.pi - overlap
    lons = np.concatenate((lons, lons[mask] - np.pi*2))
    lats = np.concatenate((lats, lats[mask]))
    data = np.concatenate((data, data[mask]))

    mask = lons < -np.pi + overlap
    lons = np.concatenate((lons, lons[mask] + np.pi*2))
    lats = np.concatenate((lats, lats[mask]))
    data = np.concatenate((data, data[mask]))

    tri = mtri.Triangulation(lons, lats)

    if refine:
        refiner = mtri.UniformTriRefiner(tri)
        tri, data = refiner.refine_field(
            data,
            triinterpolator=mtri.LinearTriInterpolator(tri, data),
            subdiv=3)

    if interpolator == 'linear':
        interpolator = mtri.LinearTriInterpolator(tri, data)
    elif interpolator == 'cubic':
        interpolator = mtri.CubicTriInterpolator(tri, data, kind='mind_E')
    else:
        raise ValueError("Please give a valid interpolation method.")

    return interpolator


def _balloon_color_data(tri, data, itype):
    """Return the data array that is to be mapped to the colormap of the
    balloon.

    Parameters
    ----------
    tri : Triangulation
        The matplotlib triangulation for the sphere
    data : ndarray, double, complex double
        The data array
    itype : 'magnitude', 'phase', 'amplitude'
        What type of data should be extracted. Either the 'magnitude', 'phase',
        or 'amplitude' of the data array is used for the colormap.

    Returns
    -------
    color_data : ndarray, double
        The data array for the colormap.
    vmin : double
        The minimum of the color data
    vmax : double
        The maximum of the color data


    """
    if itype == 'phase':
        cdata = np.mod(np.angle(data), 2*np.pi)
        vmin = 0
        vmax = 2*np.pi
        colors = circmean(cdata[tri.triangles], axis=1)
    elif itype == 'magnitude':
        cdata = np.abs(data)
        vmin = np.min(cdata)
        vmax = np.max(cdata)
        colors = np.mean(cdata[tri.triangles], axis=1)
    elif itype == 'amplitude':
        vmin = np.min(data)
        vmax = np.max(data)
        colors = np.mean(data[tri.triangles], axis=1)
    else:
        raise ValueError("Invalid type of data mapping.")

    return colors, vmin, vmax


def pcolor_sphere(
        coordinates,
        data,
        cmap=None,
        colorbar=True,
        limits=None,
        cmap_encoding='phase',
        ax=None,
        **kwargs):
    """Plot data on the surface of a sphere defined by the coordinate angles
    theta and phi. The data array will be mapped onto the surface of a sphere.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. By default is is ``None``,
        were the colormap is chosen according to the `cmap_encoding`
        :py:func:`spharpy.plot.phase_twilight` for ``'phase'`` and
        ``'viridis'`` for ``'magnitude'``.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    cmap_encoding : str, optional
        The information encoded in the colormap. Can be either `'phase'`
        (in radians) or `'magnitude'`. The default is `'phase'`.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object.
    **kwargs : optional
        Additional arguments passed to the plot_trisurf function.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    plot : matplotlib.trisurf
        The trisurf object created by the function.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
        >>> spharpy.plot.pcolor_sphere(coords, data, cmap_encoding='phase')

    """
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    if cmap_encoding not in ['phase', 'magnitude']:
        raise ValueError(
            "cmap_encoding must be either 'phase' or 'magnitude'.")


    tri, xyz = _triangulation_sphere(coordinates, np.ones_like(data))
    fig = plt.gcf()

    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection='3d')

    elif '3d' not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    if cmap_encoding == 'phase':
        if cmap is None:
            cmap = phase_twilight()
        clabel = 'Phase (rad)'
    elif cmap_encoding == 'magnitude':
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        clabel = 'Magnitude'

    cdata, vmin, vmax = _balloon_color_data(tri, data, cmap_encoding)

    if limits is not None:
        vmin, vmax = limits

    plot = ax.plot_trisurf(tri,
                           xyz[2],
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax,
                           **kwargs)

    plot.set_array(cdata)

    if colorbar:
        fig.colorbar(plot, ax=ax, label=clabel)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    ax.set_box_aspect([
        np.ptp(coordinates.x),
        np.ptp(coordinates.y),
        np.ptp(coordinates.z)])

    return (ax, plot)


def balloon_wireframe(
        coordinates,
        data,
        cmap=None,
        colorbar=True,
        limits=None,
        cmap_encoding='phase',
        ax=None,
        **kwargs):
    """Plot data on a sphere defined by the coordinate angles
    theta and phi. The magnitude information is mapped onto the radius of the
    sphere. The colormap represents either the phase or the magnitude of the
    data array.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. By default is is ``None``,
        were the colormap is chosen according to the `cmap_encoding`
        :py:func:`spharpy.plot.phase_twilight` for ``'phase'`` and
        ``'viridis'`` for ``'magnitude'``.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    cmap_encoding : str, optional
        The information encoded in the colormap. Can be either `'phase'`
        (in radians) or `'magnitude'`. The default is `'phase'`.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object.
    **kwargs : optional
        Additional arguments passed to the plot_trisurf function.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    plot : matplotlib.trisurf
        The trisurf object created by the function.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
        >>> spharpy.plot.balloon_wireframe(coords, data, cmap_encoding='phase')

    """
    # input checks
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    if cmap_encoding not in ['phase', 'magnitude']:
        raise ValueError(
            "cmap_encoding must be either 'phase' or 'magnitude'.")

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    tri, xyz = _triangulation_sphere(coordinates, data)
    fig = plt.gcf()

    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection='3d')

    elif '3d' not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    if cmap_encoding == 'phase':
        if cmap is None:
            cmap = phase_twilight()
        clabel = 'Phase (rad)'
    elif cmap_encoding == 'magnitude':
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        clabel = 'Magnitude'

    cdata, vmin, vmax = _balloon_color_data(tri, data, cmap_encoding)

    if limits is not None:
        vmin, vmax = limits

    plot = ax.plot_trisurf(tri,
                           xyz[2],
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax,
                           **kwargs)

    cnorm = plt.Normalize(vmin, vmax)

    cmap_colors = cmap(cnorm(cdata))

    cmappable = mpl.cm.ScalarMappable(cnorm, cmap)
    cmappable.set_array(np.linspace(vmin, vmax, cdata.size))

    plot.set_edgecolors(cmap_colors)
    plot.set_facecolors(np.ones(cmap_colors.shape)*0.9)

    if colorbar:
        fig.colorbar(cmappable, ax=ax, label=clabel)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    plot.set_facecolors(np.ones(cmap_colors.shape)*0.9)

    ax.set_box_aspect([
        np.ptp(xyz[0]),
        np.ptp(xyz[1]),
        np.ptp(xyz[2])])

    plot.set_facecolor([0.9, 0.9, 0.9, 0.9])

    return (ax, plot)


def balloon(
        coordinates,
        data,
        cmap=None,
        colorbar=True,
        limits=None,
        cmap_encoding='phase',
        ax=None,
        **kwargs):
    """Plot data on a sphere defined by the coordinate angles theta and phi.
    The magnitude information is mapped onto the radius of the sphere.
    The colormap represents either the phase or the magnitude of the
    data array.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. By default is is ``None``,
        were the colormap is chosen according to the `cmap_encoding`
        :py:func:`spharpy.plot.phase_twilight` for ``'phase'`` and
        ``'viridis'`` for ``'magnitude'``.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    cmap_encoding : str, optional
        The information encoded in the colormap. Can be either `'phase'`
        (in radians) or `'magnitude'`. The default is `'phase'`.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object.
    **kwargs : optional
        Additional arguments passed to the plot_trisurf function.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    plot : matplotlib.trisurf
        The trisurf object created by the function.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
        >>> spharpy.plot.balloon(coords, data, cmap_encoding='phase')

    """
    # input checks
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    if cmap_encoding not in ['phase', 'magnitude']:
        raise ValueError(
            "cmap_encoding must be either 'phase' or 'magnitude'.")

    tri, xyz = _triangulation_sphere(coordinates, data)
    fig = plt.gcf()

    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection='3d')

    elif '3d' not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    if cmap_encoding == 'phase':
        if cmap is None:
            cmap = phase_twilight()
        clabel = 'Phase (rad)'
    elif cmap_encoding == 'magnitude':
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        clabel = cmap_encoding.title()

    cdata, vmin, vmax = _balloon_color_data(tri, data, cmap_encoding)

    if limits is not None:
        vmin, vmax = limits

    plot = ax.plot_trisurf(tri,
                           xyz[2],
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax,
                           **kwargs)

    plot.set_array(cdata)

    ax.set_box_aspect([
        np.ptp(xyz[0]),
        np.ptp(xyz[1]),
        np.ptp(xyz[2])])

    if colorbar:
        fig.colorbar(plot, ax=ax, label=clabel)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    return (ax, plot)


def voronoi_cells_sphere(sampling, round_decimals=13, ax=None):
    """Plot the Voronoi cells of a Voronoi tesselation on a sphere.

    Parameters
    ----------
    sampling : pyfar.Coordinates, spharpy.SamplingSphere
        Sampling as SamplingSphere object
    round_decimals : int
        Decimals to be rounded to for eliminating duplicate points in
        the voronoi diagram
    ax : AxesSubplot, None, optional
        The subplot axes to use for plotting. The used projection needs to be
        '3d'.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> coords = spharpy.samplings.gaussian(n_max=5)
        >>> spharpy.plot.voronoi_cells_sphere(coords)

    """
    if not isinstance(sampling, pf.Coordinates):
        raise ValueError("sampling must be a coordinates object.")

    sv = spherical_voronoi(sampling, round_decimals=round_decimals)
    sv.sort_vertices_of_regions()
    points = sampling.cartesian.T

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection='3d')

    if '3d' not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    ax.scatter(points[0], points[1], points[2], c='r')

    for region in sv.regions:
        polygon = Poly3DCollection(
            [sv.vertices[region]], alpha=0.5, facecolor=None)
        polygon.set_edgecolor((0, 0, 0, 1))
        polygon.set_facecolor((1, 1, 1, 0.))

        ax.add_collection3d(polygon)

    ax.set_box_aspect([
        np.ptp(sampling.x),
        np.ptp(sampling.y),
        np.ptp(sampling.z)])

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    return ax


def _combined_contour(x, y, data, limits, cmap, levels, ax):
    """Combine a filled contour plot with a black line contour plot for
    better highlighting.

    Parameters
    ----------
    x : ndarray, double
        The x coordinates.
    y : ndarray, double
        The y coordinates.
    data : ndarray, double
        The data array.
    limits : tuple, list
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped.
    cmap : :py:class:`matplotlib.colors.Colormap`
        Colormap for the plot, see matplotlib.cm.
    levels : int or array-like
        Determines the number and positions of the contours.
        If an int n, use :py:class:`matplotlib.ticker.MaxNLocator`,
        which tries to automatically choose
        no more than n+1 contour levels between minimum and maximum
        numeric values of the plot data. If array-like, draw contour lines at
        the specified levels. The values must be in increasing order.
    ax : matplotlib.axes
        The axes object into which the contour is plotted

    Returns
    -------
    cf : matplotlib.ScalarMappable
        The scalar mappable for the contour plot

    """
    extend = 'neither'
    if limits is None:
        limits = (data.min(), data.max())
    else:
        mask_min = data < limits[0]
        data[mask_min] = limits[0]
        mask_max = data > limits[1]
        data[mask_max] = limits[1]
        if np.any(mask_max) & np.any(mask_min):
            extend = 'both'
        elif np.any(mask_max) & ~np.any(mask_min):
            extend = 'max'
        elif ~np.any(mask_max) & np.any(mask_min):
            extend = 'min'

    ax.tricontour(x, y, data, levels=levels, linewidths=0.5, colors='k',
                  vmin=limits[0], vmax=limits[1], extend=extend)
    return ax.tricontourf(
        x, y, data, levels=levels, cmap=cmap, vmin=limits[0], vmax=limits[1],
        extend=extend)


def pcolor_map(
        coordinates,
        data,
        cmap='viridis',
        colorbar=True,
        limits=None,
        projection='mollweide',
        refine=False,
        ax=None,
        **kwargs):
    """
    Plot the map projection of data points sampled on a spherical surface.
    The data has to be real.

    Notes
    -----
    In case limits are given, all out of bounds data will be clipped to the
    respective limit.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : numpy.ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. Default is 'viridis'.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    projection : str, optional
        The projection of the map. Default is 'mollweide'. See
        :py:doc:`matplotlib:gallery/subplots_axes_and_figures/geo_demo`
        for more information on available projections in matplotlib.
    refine : bool, optional
        Whether to refine the triangulation before plotting.
        Default is `False`.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object with the specified projection.
    **kwargs : optional
        Additional arguments passed to the tripcolor function.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    cf : matplotlib.tri.TriContourSet
        The contour plot object.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(2*coords.colatitude) * np.cos(2*coords.azimuth)
        >>> spharpy.plot.pcolor_map(coords, data)

    """
    # input checks
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    if not isinstance(refine, bool):
        raise ValueError("refine must be a boolean.")

    height, latitude, longitude = coordinates2latlon(coordinates)
    tri = mtri.Triangulation(longitude, latitude)
    if refine is not None:
        subdiv = refine if isinstance(refine, int) else 2
        refiner = mtri.UniformTriRefiner(tri)
        tri, data = refiner.refine_field(
            data,
            triinterpolator=mtri.LinearTriInterpolator(tri, data),
            subdiv=subdiv)

    fig = plt.gcf()

    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection=projection)

    if ax.name != projection:
        raise ValueError(
            f"The projection of the axis needs to be '{projection}'"
            f", but is '{ax.name}'")

    ax.set_xlabel('Longitude [$^\\circ$]')
    ax.set_ylabel('Latitude [$^\\circ$]')

    extend = 'neither'
    if limits is None:
        limits = (data.min(), data.max())
    else:
        mask_min = data < limits[0]
        data[mask_min] = limits[0]
        mask_max = data > limits[1]
        data[mask_max] = limits[1]
        if np.any(mask_max) & np.any(mask_min):
            extend = 'both'
        elif np.any(mask_max) & ~np.any(mask_min):
            extend = 'max'
        elif ~np.any(mask_max) & np.any(mask_min):
            extend = 'min'

    cf = ax.tripcolor(
        tri, data, cmap=cmap, vmin=limits[0], vmax=limits[1], **kwargs)

    plt.grid(True)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax, extend=extend)
        cb.set_label('Amplitude')

    return ax, cf


def contour_map(
        coordinates,
        data,
        cmap='viridis',
        colorbar=True,
        limits=None,
        projection='mollweide',
        levels=None,
        ax=None):
    """
    Plot the map projection of data points sampled on a spherical surface.
    The data has to be real.

    Notes
    -----
    In case limits are given, all out of bounds data will be clipped to the
    respective limit.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. Default is 'viridis'.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    projection : str, optional
        The projection of the map. Default is 'mollweide'. See
        :py:doc:`matplotlib:gallery/subplots_axes_and_figures/geo_demo`
        for more information on available projections in matplotlib.
    levels : int or array-like, optional
        Determines the number and positions of the contours.
        If an int n, use :py:class:`matplotlib.ticker.MaxNLocator`,
        which tries to automatically choose
        no more than n+1 contour levels between minimum and maximum
        numeric values of the plot data. If array-like, draw contour lines at
        the specified levels. The values must be in increasing order.
        Default is ``None``, the levels are chosen automatically by
        Matplotlib.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object with the specified projection.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    cf : matplotlib.contour.QuadContourSet
        The contour plot object.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(2*coords.colatitude) * np.cos(2*coords.azimuth)
        >>> spharpy.plot.contour_map(coords, data)

    """
    # input checks
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    data = data.copy()

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca() if fig.axes else plt.axes(projection=projection)

    if ax.name != projection:
        raise ValueError(
            f"The projection of the axis needs to be '{projection}'"
            f", but is '{ax.name}'")

    ax.set_xlabel('Longitude [$^\\circ$]')
    ax.set_ylabel('Latitude [$^\\circ$]')

    _, latitude, longitude = coordinates2latlon(coordinates)
    cf = _combined_contour(longitude, latitude, data, limits, cmap, levels, ax)

    if type(levels) is int:
        levels = mpl.ticker.MaxNLocator(levels)

    plt.grid(True)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax, ticks=levels)
        cb.set_label('Amplitude')

    return ax, cf


def contour(
        coordinates,
        data,
        cmap='viridis',
        colorbar=True,
        limits=None,
        levels=None,
        ax=None):
    """
    Plot the map projection of data points sampled on a spherical surface.
    The data has to be real-valued.

    Notes
    -----
    In case limits are given, all out of bounds data will be clipped to the
    respective limit.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data: ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`, optional
        Colormap for the plot, see matplotlib.cm. Default is 'viridis'.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    levels : int or array-like, optional
        Determines the number and positions of the contours.
        If an int n, use :py:class:`matplotlib.ticker.MaxNLocator`,
        which tries to automatically choose
        no more than n+1 contour levels between minimum and maximum
        numeric values of the plot data. If array-like, draw contour lines at
        the specified levels. The values must be in increasing order.
        Default is ``None``, the levels are chosen automatically by
        Matplotlib.
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default `None`, which
        will create a new axis object with the specified projection.

    Returns
    -------
    ax : matplotlib.axis
        The axis object used for plotting.
    cf : matplotlib.contour.QuadContourSet
        The contour plot object.

    Examples
    --------

    .. plot::

        >>> import spharpy
        >>> import numpy as np
        >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
        >>> data = np.sin(2*coords.colatitude) * np.cos(2*coords.azimuth)
        >>> spharpy.plot.contour(coords, data)

    """
    # input checks
    _check_input_parameters(coordinates, data, cmap, colorbar, limits)
    data = data.copy()

    _, latitude, longitude = coordinates2latlon(coordinates)
    lat_deg = latitude * 180/np.pi
    lon_deg = longitude * 180/np.pi
    fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    if ax.name != 'rectilinear':
        raise ValueError(
            f"The projection of the axis needs to be 'rectilinear'"
            f", but is '{ax.name}'")

    ax.set_xlabel('Longitude [$^\\circ$]')
    ax.set_ylabel('Latitude [$^\\circ$]')

    cf = _combined_contour(lon_deg, lat_deg, data, limits, cmap, levels, ax)

    if type(levels) is int:
        levels = mpl.ticker.MaxNLocator(levels)

    plt.grid(True)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax, ticks=levels)
        cb.set_label('Amplitude')

    return ax, cf


class MidpointNormalize(colors.Normalize):
    """Colormap norm with a defined midpoint. Useful for normalization of
    colormaps representing deviations from a defined midpoint.
    Taken from the official matplotlib documentation at
    https://matplotlib.org/users/colormapnorms.html.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=0., clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):  # noqa: ARG002
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def coordinates2latlon(coords: pf.Coordinates):
    r"""Transforms from Cartesian coordinates to Geocentric coordinates.

    .. math::

        h = \sqrt{x^2 + y^2 + z^2},

        \theta = \pi/2 - \arccos(\frac{z}{r}),

        \phi = \arctan(\frac{y}{x})

        -\pi/2 < \theta < \pi/2,

        -\pi < \phi < \pi

    where :math:`h` is the heigth, :math:`\theta` is the latitude angle
    and :math:`\phi` is the longitude angle

    Parameters
    ----------
    coords : pyfar.Coordinates, spharpy.SamplingSphere
        Cartesian Coordiantes are transformed to Geocentric coordinates

    Returns
    -------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    """
    x = coords.x
    y = coords.y
    z = coords.z
    height = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.pi/2 - np.arccos(z/height)
    longitude = np.arctan2(y, x)
    return height, latitude, longitude


def _check_input_parameters(coordinates, data, cmap, colorbar, limits):
    """Check the input parameters for the plotting functions.

    The function raises ValueError if the input parameters are not valid.

    Parameters
    ----------
    coordinates : pyfar.Coordinates, spharpy.SamplingSphere
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : str, :py:class:`matplotlib.colors.Colormap`
        Colormap for the plot, see matplotlib.cm.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is `True`.
    limits : tuple, list, optional
        Tuple or list containing the maximum and minimum to which the colormap
        needs to be clipped. If `None`, the limits are set to the minimum and
        maximum of the data.
    """
    if not isinstance(colorbar, bool):
        raise ValueError("colorbar must be a boolean.")
    if not isinstance(cmap, (str, type(None), mpl.colors.Colormap)):
        raise ValueError(
            "cmap must be a string, Colormap object, or None.")
    if not isinstance(coordinates, pf.Coordinates):
        raise ValueError("coordinates must be a coordinates object.")
    if not isinstance(data, np.ndarray):
        raise ValueError(
            "data must be a 1D array with the same cshape as the coordinates.")
    if data.shape[-1] != coordinates.cshape[-1]:
        raise ValueError(
            "data must be a 1D array with the same cshape as the coordinates.")
    if limits is not None and not isinstance(limits, (tuple, list)):
        raise ValueError(
            "limits must be a tuple or list containing the minimum and "
            "maximum values for the colormap or None.")
    if limits is not None and len(limits) != 2:
        raise ValueError(
            "limits must be a tuple or list containing the minimum and "
            "maximum values for the colormap or None.")
