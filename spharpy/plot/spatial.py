"""
Plot functions for spatial data
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri
import scipy.spatial as sspat
from matplotlib import cm
from spharpy.samplings import sph2cart

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


def scatter(x, y, z):
    """Plot the x, y, and z coordinates of the sampling grid in the 3d space.

    Parameters
    ----------
    x : ndarray, double
        x - coordinates
    y : ndarray, double
        y - coordinates
    z : ndarray, double
        z - coordinates

    """
    fig = plt.gcf()
    ax = fig.add_subplot('111', projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    set_aspect_equal_3d(ax)
    plt.show()


def surf(theta, phi, data, cmap=cm.viridis, phase=False):
    """Plot data on the surface of a sphere defined by the coordinate angles
    theta and phi

    Parameters
    ----------
    theta : ndarray, double
        Elevation angles
    phi : ndarray, double
        Azimuth angles
    data : ndarray, double
        Data for each angle, must be of the same dimension as theta and phi
    cmap : matplotlib colomap, optional
        Colormap for the plot, see matplotlib.cm
    phase : boolean, optional
        Decode the phase of the data in the colormap. This option will be activated
        by default of the data is complex valued.
    """
    x, y, z = sph2cart(np.abs(data), theta, phi)
    hull = sspat.ConvexHull(np.asarray(sph2cart(np.ones(theta.shape[0]), theta, phi)).T)
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')

    if np.iscomplex(data).any() or phase:
        cmap = cm.hsv
        phase_colors = cmap(np.mod(np.angle(data), 2*np.pi)/(2*np.pi))
        facecolors = np.mean(phase_colors[tri.triangles], axis=1)
        plot = ax.plot_trisurf(tri, z, antialiased=True)
        plot.set_facecolors(facecolors)
    else:
        plot = ax.plot_trisurf(tri, z, antialiased=True, cmap=cmap)
        fig.colorbar(plot, shrink=0.75, aspect=20)

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    set_aspect_equal_3d(ax)
    plt.show()


def contour(latitude, longitude, data, limits=None, cmap=cm.viridis):
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
        Data for each angle, must be of the same dimension as latitude and longitude

    """
    lat_deg = latitude * 180/np.pi
    lon_deg = longitude * 180/np.pi
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

    plt.show()
