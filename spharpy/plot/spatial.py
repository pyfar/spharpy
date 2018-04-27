"""
Plot functions for spatial data
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri
import scipy.spatial as sspat
from matplotlib import cm
import matplotlib

from spharpy.samplings import sph2cart
from spharpy.samplings.coordinates import Coordinates


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
    ax = fig.add_subplot('111', projection='3d')
    ax.scatter(coordinates.x, coordinates.y, coordinates.z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    set_aspect_equal_3d(ax)
    plt.show()


def balloon(coordinates, data, cmap=cm.viridis, phase=False, show=True):
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
    ax = fig.add_subplot(111, projection='3d', aspect='equal')

    if np.iscomplex(data).any() or phase:
        cdata = np.mod(np.angle(data), 2*np.pi)
        cmap = cm.hsv
        vmin = 0
        vmax = 2*np.pi
    else:
        cdata = np.abs(data)
        vmin = np.min(cdata)
        vmax = np.max(cdata)

    plot = ax.plot_trisurf(tri,
                           z,
                           cmap=cmap,
                           antialiased=True,
                           vmin=vmin,
                           vmax=vmax)

    colors = np.mean(cdata[tri.triangles], axis=1)
    plot.set_array(colors)

    fig.colorbar(plot, shrink=0.75, aspect=20)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    set_aspect_equal_3d(ax)
    if show:
        plt.show()


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
