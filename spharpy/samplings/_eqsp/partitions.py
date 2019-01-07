"""
Python library partition the unit sphere into partition with equal area.
"""

from math import gcd
import numpy as np
import scipy.special as sps


def point_set(dimension, N, **kwargs):
    """TODO: Docstring for eq_point_set.

    :dimension: TODO
    :N: TODO
    :**kwargs: TODO
    :returns: TODO

    """
    dimension = np.asarray(dimension)
    N = np.asarray(N)
    return polar2cart(point_set_polar(dimension, N))


def point_set_polar(dimension, N, **kwargs):
    """TODO: Docstring for point_set_polar.

    :dimension: TODO
    :returns: TODO

    """
    a_cap, n_regions = caps(dimension, N)

    if dimension == 1:
        points_s = a_cap - np.pi/N
    else:
        # import ipdb; ipdb.set_trace()
        n_collars = np.size(n_regions) - 2
        use_cache = dimension >= 2
        if use_cache:
            cache_size = np.floor(n_collars/2)
            # cache =

        points_s = np.zeros((dimension, N))
        point_n = 2
        # points = np.zeros((dimension, N))


        # TODO check what the offset is for
        offset = 0

        for collar_n in range(0, n_collars):
            # a_top is the colatitude of the top of the current collar.
            a_top = a_cap[collar_n]

            # a_bot is the colatitude of the bottom of the current collar.
            a_bot = a_cap[collar_n+1]

            # n_in_collar is the number of regions in the current collar.
            n_in_collar = n_regions[collar_n+1]

            # if use_cache:
            #     twin_collar_n = n_collars - collar_n
            #
            #     # if twin_collar_n <= cache_size && ...
            #     #     size(cache{twin_collar_n},2) == n_in_collar
            #     #     points_1 = cache{twin_collar_n};
            #     # else
            #     points_l = point_set_polar(dimension - 1, n_in_collar)
            # else:
            points_l = point_set_polar(dimension - 1, n_in_collar)

            a_point = (a_top + a_bot)/2

            point_l_n = np.arange(0, np.size(points_l), dtype=np.int)
            # points_l = points_l[np.newaxis]

            if dimension == 2:
                points_s[0:dimension-1, point_n+point_l_n-1] = \
                    np.mod(points_l[point_l_n] + 2*np.pi*offset, 2*np.pi)

                offset += circle_offset(n_in_collar, n_regions[2+collar_n])
                offset -= np.floor(offset)
            else:
                points_s[0:dimension-2, point_n+point_l_n-1] = points_l[:, point_l_n]

            # import ipdb; ipdb.set_trace()

            points_s[dimension-1, point_n+point_l_n-1] = a_point
            # point_n = point_n + points_l.shape[1]
            point_n += np.size(points_l)

        points_s[:, -1] = np.zeros(dimension)
        points_s[-1, -1] = np.pi


    return points_s

def caps(dimension, N):
    """TODO: Docstring for cap.

    :dimension: TODO
    :N: TODO
    :returns: TODO

    """
    if dimension == 1:
        s_cap = np.linspace(2*np.pi/N, 2*np.pi, N)
        n_regions = np.ones(10, dtype=np.int)
    elif N == 1:
        s_cap = np.pi
        n_regions = 1
    else:
        c_polar = polar_colat(dimension, N)
        n_collars = num_collars(N, c_polar, ideal_collar_angle(dimension, N))
        r_regions = ideal_region_list(dimension, N, c_polar, n_collars)
        n_regions = round_to_naturals(N, r_regions)
        s_cap = cap_colats(dimension, N, c_polar, n_regions)

    return s_cap, n_regions


def polar_colat(dimension, N):
    """TODO: Docstring for polar_colat.
    :returns: TODO

    """
    # enough = N > 2
    # c_polar = np.empty()
    if N == 1:
        c_polar = np.pi
    elif N == 2:
        c_polar = np.pi/2
    else:
        ideal_region_area = area_of_ideal_region(dimension, N)
        c_polar = sradius_of_cap(dimension, ideal_region_area)

    # c_polar[N == 1] = np.pi
    # c_polar[N == 2] = np.pi/2
    # ideal_region_area = equtils.area_of_ideal_region(dimension, N[enough])
    # ideal_region_area = equtils.area_of_ideal_region(dimension, N)
    # c_polar[enough] = equtils.sradius_of_cap(dimension, ideal_region_area)
    return c_polar



def ideal_region_list(dimension, N, c_polar, n_collars):
    """TODO: Docstring for ideal_region_list.

    :dimension: TODO
    :N: TODO
    :c_polar: TODO
    :n_collars: TODO
    :returns: TODO

    """
    r_regions = np.zeros(2+n_collars)
    r_regions[0] = 1

    if n_collars > 0:
        a_fitting = (np.pi - 2*c_polar) / n_collars
        ideal_region_area = area_of_ideal_region(dimension, N)
        for collar_n in range(1, n_collars+1):
            ideal_collar_area = area_of_collar(dimension,
                                                       c_polar + (collar_n -1) * a_fitting,
                                                       c_polar + collar_n * a_fitting)
            r_regions[collar_n] = ideal_collar_area / ideal_region_area

    r_regions[-1] = 1

    return r_regions

def round_to_naturals(N, r_regions):
    """TODO: Docstring for round_to_naturals.

    :N: TODO
    :r_regions: TODO
    :returns: TODO

    """
    r_regions = np.asarray(r_regions)
    n_regions = np.zeros(r_regions.shape, dtype=np.int)
    discrepancy = 0

    for zone_n in range(0, np.size(r_regions)):
        n_regions[zone_n] = np.rint(r_regions[zone_n] + discrepancy)
        discrepancy += (r_regions[zone_n] - n_regions[zone_n])

    return n_regions

def cap_colats(dimension, N, c_polar, n_regions):
    """TODO: Docstring for cap_colats.

    :dimension: TODO
    :N: TODO
    :c_polar: TODO
    :n_regions: TODO
    :returns: TODO

    """
    c_caps = np.zeros(np.size(n_regions))
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dimension, N)
    n_collars = np.size(n_regions) - 2
    subtotal_n_regions = 1

    for collar_n in range(1, n_collars+1):
        subtotal_n_regions += n_regions[collar_n]
        c_caps[collar_n] = sradius_of_cap(dimension,
                                                  subtotal_n_regions*ideal_region_area)

    c_caps[-1] = np.pi
    return c_caps

def num_collars(N, c_polar, a_ideal):
    """TODO: Docstring for _num_collars.

    :N: TODO
    :c_polar: TODO
    :a_ideal: TODO
    :returns: TODO

    """
    # N = np.asarray(N)
    # a_ideal = np.asarray(a_ideal)
    # c_polar = np.asarray(c_polar)
    # n_collars = np.zeros(N.size)
    # mask = np.int((N > 2) and (a_ideal > 0))
    # n_collars[mask] = np.max(1, np.round((np.pi - 2*c_polar[mask])) / a_ideal[mask])
    if np.int((N > 2) and (a_ideal > 0)):
        n_collars = np.maximum(1, np.rint((np.pi - 2*c_polar) / a_ideal))
    else:
        n_collars = 0

    return np.int(n_collars)

def circle_offset(n_top, n_bot, extra_twist=False):
    """TODO: Docstring for circle_offset.

    :n_top: TODO
    :n_bot: TODO
    :extra_twist: TODO
    :returns: TODO

    """
    offset = (1/n_bot - 1/n_top)/2 + gcd(n_top, n_bot) / (2*n_top*n_bot)
    if extra_twist:
        twist = 6
        offset += twist/n_bot

    return offset


def ideal_collar_angle(dimension, N):
    """TODO: Docstring for ideal_collar_angle.

    :dimension: TODO
    :N: TODO
    :returns: TODO

    """
    return area_of_ideal_region(dimension, N)**(1 / dimension)


def area_of_ideal_region(dimension, N):
    """TODO: Docstring for area_of_ideal_region.

    :dimension: TODO
    :N: TODO
    :returns: TODO

    """
    return area_of_sphere(dimension)/N


def area_of_sphere(dimension):
    """TODO: Docstring for area_of_sphere.

    :dimension: TODO
    :returns: TODO

    """
    power = (dimension + 1)/2
    return 2*np.pi**power / sps.gamma(power)

def area_of_collar(dimension, a_top, a_bot):
    """TODO: Docstring for area_of_collar.

    :dimension: TODO
    :a_top: TODO
    :a_bot: TODO
    :returns: TODO

    """
    return area_of_cap(dimension, a_bot) - area_of_cap(dimension, a_top)

def area_of_cap(dimension, s_cap):
    """TODO: Docstring for area_of_cap.

    :dimension: TODO
    :s_cap: TODO
    :returns: TODO

    """
    if dimension == 1:
        area = 2 * s_cap
    elif dimension == 2:
        area = 4 * np.pi * np.sin(s_cap / 2)**2
    elif dimension == 3:
        # TODO
        pass
    else:
        area = area_of_sphere(dimension) * sps.betainc(np.sin(s_cap/2)**2,
                                                       dimension/2,
                                                       dimension/2)

    return area

def sradius_of_cap(dimension, area):
    """TODO: Docstring for sradius_of_cap.

    :dimension: TODO
    :area: TODO
    :returns: TODO

    """
    if dimension == 1:
        s_cap = area/2
    elif dimension == 2:
        s_cap = 2*np.arcsin(np.sqrt(area / np.pi) / 2)
    else:
        raise NotImplementedError
    return np.asarray(s_cap)

def polar2cart(points_polar):
    """Comnversion from the polar angles theta and phi to Cartesian coordinates

        x = cos(phi) * sin(theta)
        y = sin(phi) * sin(theta)
        x = cos(theta)

    :points_polar: TODO
    :returns: TODO

    """
    points_cart = np.zeros((points_polar.shape[0]+1, points_polar.shape[1]))
    points_cart[0, :] = np.cos(points_polar[0, :]) * np.sin(points_polar[1, :])
    points_cart[1, :] = np.sin(points_polar[0, :]) * np.sin(points_polar[1, :])
    points_cart[2, :] = np.cos(points_polar[1, :])
    return points_cart
