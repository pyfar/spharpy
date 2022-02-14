"""
Plot functions for spatial data
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.spatial as sspat
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
__all__ = [Axes3D]
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packaging import version
from scipy.stats import circmean

from .cmap import phase_twilight

from spharpy.samplings import sph2cart, spherical_voronoi


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


def scatter(coordinates):
    """Plot the x, y, and z coordinates of the sampling grid in the 3d space.

    Parameters
    ----------
    coordinates : Coordinates

    """
    fig = plt.gcf()
    if 'Axes3D' in fig.axes.__str__():
        ax = plt.gca()
    else:
        ax = plt.gca(projection='3d')

    ax.scatter(coordinates.x, coordinates.y, coordinates.z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    set_aspect_equal_3d(ax)
    plt.show()


def _triangulation_sphere(sampling, data):
    """Triangulation for data points sampled on a spherical surface.

    Parameters
    ----------
    sampling : Coordinates
        Coordinate object for which the triangulation is calculated
    data : array, shape(n_points)
        Sampled data

    Returns
    -------
    triangulation : matplotlib Triangulation

    """
    x, y, z = sph2cart(
        np.abs(data),
        sampling.elevation,
        sampling.azimuth)
    hull = sspat.ConvexHull(
        np.asarray(sph2cart(
            np.ones(sampling.n_points),
            sampling.elevation,
            sampling.azimuth)).T)
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)

    return tri, z


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
    sampling : Coordinates
        The coordinates at which the data is sampled.
    data : ndarray, double
        The sampled data points.
    overlap : float, (pi/4)
        The overlap for the periodic extension in azimuth angle, given in
        radians
    refine : bool (False)
        Refine the mesh before interpolating
    interpolator : linear, cubic
        The interpolation method to be used

    Returns
    -------
    interp : LinearTriInterpolator, CubicTriInterpolator
        The interpolator object.

    Note
    ----
    Internally, matplotlibs LinearTriInterpolator or CubicTriInterpolator
    are used.

    """
    lats = sampling.latitude
    lons = sampling.longitude

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
        Whether to plot magnitude levels or the phase.

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
        show=True,
        phase=False,
        *args,
        **kwargs):
    """Plot data on the surface of a sphere defined by the coordinate angles
    theta and phi. The data array will be mapped onto the surface of a sphere.

    Note
    ----
    When plotting the phase encoded in the colormap, the function will switch
    to the HSV colormap and ignore the user input for the cmap input variable.

    Parameters
    ----------
    coordinates : Coordinates
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : matplotlib colomap, optional
        Colormap for the plot, see matplotlib.cm
    phase : boolean, optional
        Encode the phase of the data in the colormap. This option will be
        activated by default of the data is complex valued.
    show : boolean, optional
        Whether to show the figure or not

    """
    tri, z = _triangulation_sphere(coordinates, np.ones_like(data))
    fig = plt.gcf()

    if colorbar:
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 0.05],
            height_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        cax = fig.add_subplot(gs[0, 1])
    else:
        if 'Axes3D' in fig.axes.__str__():
            ax = plt.gca()
        else:
            ax = plt.gca(projection='3d')

    if np.iscomplex(data).any() or phase:
        itype = 'phase'
        if cmap is None:
            cmap = phase_twilight()
    else:
        itype = 'amplitude'
        if cmap is None:
            cmap = cm.viridis

    cdata, vmin, vmax = _balloon_color_data(tri, data, itype)

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    plot = ax.plot_trisurf(tri,
                           z,
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax)

    plot.set_array(cdata)

    set_aspect_equal_3d(ax)

    if colorbar:
        plt.colorbar(plot, cax=cax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    ax.set_proj_type('ortho')

    if show:
        plt.show()

    return plot


def balloon_wireframe(
        coordinates,
        data,
        cmap=None,
        phase=False,
        show=True,
        colorbar=True):
    """Plot data on a sphere defined by the coordinate angles
    theta and phi. The magnitude information is mapped onto the radius of the
    sphere. The colormap represents either the phase or the magnitude of the
    data array.

    Note
    ----
    When plotting the phase encoded in the colormap, the function will switch
    to the HSV colormap and ignore the user input for the cmap input variable.

    Parameters
    ----------
    coordinates : Coordinates
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : matplotlib colomap, optional
        Colormap for the plot, see matplotlib.cm
    phase : boolean, optional
        Encode the phase of the data in the colormap. This option will be
        activated by default of the data is complex valued.
    show : boolean, optional
        Whether to show the figure or not
    """
    tri, z = _triangulation_sphere(coordinates, data)
    fig = plt.gcf()

    if colorbar:
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 0.05],
            height_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        cax = fig.add_subplot(gs[0, 1])
    else:
        if 'Axes3D' in fig.axes.__str__():
            ax = plt.gca()
        else:
            ax = plt.gca(projection='3d')

    if np.iscomplex(data).any() or phase:
        itype = 'phase'
        if cmap is None:
            cmap = phase_twilight()
    else:
        itype = 'magnitude'
        if cmap is None:
            cmap = cm.viridis

    cdata, vmin, vmax = _balloon_color_data(tri, data, itype)

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    plot = ax.plot_trisurf(tri,
                           z,
                           # cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax)

    cnorm = plt.Normalize(vmin, vmax)
    cmap_colors = cmap(cnorm(cdata))

    cmappable = mpl.cm.ScalarMappable(cnorm, cmap)
    cmappable.set_array(np.linspace(vmin, vmax, cdata.size))

    plot.set_edgecolors(cmap_colors)

    set_aspect_equal_3d(ax)
    plot.set_facecolors(np.ones(cmap_colors.shape)*0.9)

    if colorbar:
        plt.colorbar(cmappable, cax=cax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    plot.set_facecolors(np.ones(cmap_colors.shape)*0.9)
    ax.set_proj_type('ortho')

    if show:
        plt.show()

    plot.set_facecolor([0.9, 0.9, 0.9, 0.9])

    return plot


def balloon(
        coordinates,
        data,
        cmap=None,
        phase=False,
        show=True,
        colorbar=True,
        *args,
        **kwargs):
    """Plot data on a sphere defined by the coordinate angles theta and phi.
    The magnitude information is mapped onto the radius of the sphere.
    The colormap represents either the phase or the magnitude of the
    data array.


    Note
    ----
    When plotting the phase encoded in the colormap, the function will switch
    to the HSV colormap and ignore the user input for the cmap input variable.

    Parameters
    ----------
    coordinates : Coordinates
        Coordinates defining a sphere
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    cmap : matplotlib colomap, optional
        Colormap for the plot, see matplotlib.cm
    phase : boolean, optional
        Encode the phase of the data in the colormap. This option will be
        activated by default of the data is complex valued.
    show : boolean, optional
        Wheter to show the figure or not
    """
    tri, z = _triangulation_sphere(coordinates, data)
    fig = plt.gcf()

    if colorbar:
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 0.05],
            height_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        cax = fig.add_subplot(gs[0, 1])
    else:
        if 'Axes3D' in fig.axes.__str__():
            ax = plt.gca()
        else:
            ax = plt.gca(projection='3d')

    if np.iscomplex(data).any() or phase:
        itype = 'phase'
        if cmap is None:
            cmap = phase_twilight()
    else:
        itype = 'magnitude'
        if cmap is None:
            cmap = cm.viridis

    cdata, vmin, vmax = _balloon_color_data(tri, data, itype)

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    plot = ax.plot_trisurf(tri,
                           z,
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax,
                           *args,
                           **kwargs)

    plot.set_array(cdata)

    set_aspect_equal_3d(ax)

    if colorbar:
        plt.colorbar(plot, cax=cax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')

    ax.set_proj_type('ortho')

    if show:
        plt.show()

    return plot


def voronoi_cells_sphere(sampling, round_decimals=13):
    """Plot the Voronoi cells of a Voronoi tesselation on a sphere.

    Parameters
    ----------
    sampling : SamplingSphere
        Sampling as SamplingSphere object
    round_decimals : int
        Decimals to be rounded to for eliminating duplicate points in
        the voronoi diagram

    """
    sv = spherical_voronoi(sampling, round_decimals=round_decimals)
    sv.sort_vertices_of_regions()
    points = sampling.cartesian.T

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

    for region in sv.regions:
        polygon = Poly3DCollection(
            [sv.vertices[region]], alpha=0.5, facecolor=None)
        polygon.set_edgecolor((0, 0, 0, 1))
        polygon.set_facecolor((1, 1, 1, 0.))

        ax.add_collection3d(polygon)

    set_aspect_equal_3d(ax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    ax.set_proj_type('ortho')


def _combined_contour(x, y, data, limits, cmap, ax):
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
    cmap : matplotlib.colormap
        The colormap
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

    ax.tricontour(x, y, data, linewidths=0.5, colors='k',
                  vmin=limits[0], vmax=limits[1], extend=extend)
    cf = ax.tricontourf(x, y, data, cmap=cmap,
                        vmin=limits[0], vmax=limits[1], extend=extend)

    return cf


def pcolor_map(
        coordinates,
        data,
        projection='mollweide',
        limits=None,
        cmap=cm.viridis,
        show=True,
        refine=False,
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
    latitude: ndarray, double
        Geodetic latitude angle of the map, must be in [-pi/2, pi/2]
    longitude: ndarray, double
        Geodetic longitude angle of the map, must be in [-pi, pi]
    data: ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    show : boolean, optional
        Wheter to show the figure or not

    """
    tri = mtri.Triangulation(coordinates.longitude, coordinates.latitude)
    if refine is not None:
        if isinstance(refine, int):
            subdiv = refine
        else:
            subdiv = 2
        refiner = mtri.UniformTriRefiner(tri)
        tri, data = refiner.refine_field(
            data,
            triinterpolator=mtri.LinearTriInterpolator(tri, data),
            subdiv=subdiv)

    fig = plt.gcf()

    ax = plt.axes(projection=projection)

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
    cb = fig.colorbar(cf, ax=ax, extend=extend)
    cb.set_label('Amplitude')
    if show:
        plt.show()

    return cf


def contour_map(
        coordinates,
        data,
        projection='mollweide',
        limits=None,
        cmap=cm.viridis,
        colorbar=True,
        show=True,
        levels=None):
    """
    Plot the map projection of data points sampled on a spherical surface.
    The data has to be real.

    Notes
    -----
    In case limits are given, all out of bounds data will be clipped to the
    respective limit.

    Parameters
    ----------
    latitude: ndarray, double
        Geodetic latitude angle of the map, must be in [-pi/2, pi/2]
    longitude: ndarray, double
        Geodetic longitude angle of the map, must be in [-pi, pi]
    data: ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    show : boolean, optional
        Wheter to show the figure or not

    """
    fig = plt.gcf()

    res = int(np.ceil(np.sqrt(coordinates.n_points)))

    xi, yi = np.meshgrid(
        np.linspace(-np.pi, np.pi, res*2),
        np.linspace(-np.pi/2, np.pi/2, res))

    interp = interpolate_data_on_sphere(coordinates, data)
    zi = interp(xi, yi)

    # ax = plt.axes(projection=projection)
    ax = plt.gca(projection=projection)

    ax.set_xlabel('Longitude [$^\\circ$]')
    ax.set_ylabel('Latitude [$^\\circ$]')

    extend = 'neither'
    if limits is None:
        limits = (zi.min(), zi.max())
    else:
        mask_min = zi < limits[0]
        zi[mask_min] = limits[0]
        mask_max = zi > limits[1]
        zi[mask_max] = limits[1]
        if np.any(mask_max) and np.any(mask_min):
            extend = 'both'
        elif np.any(mask_max) and not np.any(mask_min):
            extend = 'max'
        elif not np.any(mask_max) and np.any(mask_min):
            extend = 'min'

    ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k',
               vmin=limits[0], vmax=limits[1], extend=extend)
    cf = ax.pcolormesh(xi, yi, zi, cmap=cmap, shading='gouraud',
                       vmin=limits[0], vmax=limits[1])

    plt.grid(True)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax, ticks=levels)
        cb.set_label('Amplitude')
    if show:
        plt.show()

    return cf


def contour(coordinates, data, limits=None, cmap=cm.viridis, show=True):
    """
    Plot the map projection of data points sampled on a spherical surface.
    The data has to be real.

    Notes
    -----
    In case limits are given, all out of bounds data will be clipped to the
    respective limit.

    Parameters
    ----------
    latitude: ndarray, double
        Geodetic latitude angle of the map, must be in [-pi/2, pi/2]
    longitude: ndarray, double
        Geodetic longitude angle of the map, must be in [-pi, pi]
    data: ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    show : boolean, optional
        Wheter to show the figure or not

    """
    lat_deg = coordinates.latitude * 180/np.pi
    lon_deg = coordinates.longitude * 180/np.pi
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel('Longitude [$^\\circ$]')
    ax.set_ylabel('Latitude [$^\\circ$]')

    cf = _combined_contour(lon_deg, lat_deg, data, limits, cmap, ax)

    plt.grid(True)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label('Amplitude')
    if show:
        plt.show()

    return cf


class MidpointNormalize(colors.Normalize):
    """Colormap norm with a defined midpoint. Useful for normalization of
    colormaps representing deviations from a defined midpoint.
    Taken from the official matplotlib documentation at
    https://matplotlib.org/users/colormapnorms.html
    """
    def __init__(self, vmin=None, vmax=None, midpoint=0., clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
