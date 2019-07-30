"""
Plot functions for spatial data
"""

from packaging import version

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

import scipy.spatial as sspat
from scipy.stats import circmean

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


def balloon(coordinates, data, cmap=cm.viridis, phase=False, show=True,
            colorbar=True):
    """Plot data on the surface of a sphere defined by the coordinate angles
    theta and phi

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
    n_points = coordinates.n_points
    x, y, z = sph2cart(np.abs(data),
                       coordinates.elevation,
                       coordinates.azimuth)
    hull = sspat.ConvexHull(np.asarray(sph2cart(np.ones(n_points),
                                                coordinates.elevation,
                                                coordinates.azimuth)).T)
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)
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
        cdata = np.mod(np.angle(data), 2*np.pi)
        cmap = cm.hsv
        vmin = 0
        vmax = 2*np.pi
        colors = circmean(cdata[tri.triangles], axis=1)
    else:
        cdata = np.abs(data)
        vmin = np.min(cdata)
        vmax = np.max(cdata)
        colors = np.mean(cdata[tri.triangles], axis=1)

    if version.parse(mpl.__version__) < version.parse('3.1.0'):
        ax.set_aspect('equal')

    plot = ax.plot_trisurf(tri,
                           z,
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax)

    plot.set_array(colors)

    set_aspect_equal_3d(ax)

    if colorbar:
        plt.colorbar(plot, cax=cax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
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
        polygon = Poly3DCollection([sv.vertices[region]], alpha=0.5, facecolor=None)
        polygon.set_edgecolor((0, 0, 0, 1))
        polygon.set_facecolor((1, 1, 1, 0.))

        ax.add_collection3d(polygon)

    set_aspect_equal_3d(ax)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')


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

    ax.tricontour(lon_deg, lat_deg, data, linewidths=0.5, colors='k',
                  vmin=limits[0], vmax=limits[1], extend=extend)
    cf = ax.tricontourf(lon_deg, lat_deg, data, cmap=cmap,
                        vmin=limits[0], vmax=limits[1], extend=extend)

    plt.grid(True)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label('Amplitude')
    if show:
        plt.show()

    return cf
