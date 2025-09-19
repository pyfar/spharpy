import os
import pytest
import spharpy as sp
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
For general information on testing plot functions see
https://pyfar-gallery.readthedocs.io/en/latest/contribute/contribution_guidelines.html#testing-plot-functions

Important:
- `create_baseline` and `compare_output` must be ``False`` when pushing
  changes to pyfar.
- `create_baseline` must only be ``True`` if the behavior of a plot function
  changed. In this case it is best practice to recreate only the baseline plots
  of the plot function (plot behavior) that changed.
"""
# global parameters -----------------------------------------------------------
create_baseline = False

# file type used for saving the plots
file_type = "png"

# if true, the plots will be compared to the baseline and an error is raised
# if there are any differences. In any case, differences are written to
# output_path as images
compare_output = False

# path handling
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

if not os.path.isdir(base_path):
    os.mkdir(base_path)
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# remove old output files
for file in os.listdir(output_path):
    os.remove(os.path.join(output_path, file))

# the naming scheme of the baseline is as follows:
# <function_name>_<parameter_name>_<parameters>.png

# testing ---------------------------------------------------------------------
@pytest.mark.parametrize('function', [
    (sp.plot.scatter),
    (sp.plot.voronoi_cells_sphere)])
def test_sampling_scatter(function):
    """Test the scatter plot with default arguments."""
    coords = sp.samplings.equal_area(n_max=0, n_points=12)

    # do plotting
    filename = f'{function.__name__}_default'
    create_figure()
    function(coords)
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)

    match = 'must be a coordinates object.'
    with pytest.raises(ValueError, match=match):
        function('coords')


@pytest.mark.parametrize('function', [
    (sp.plot.balloon),
    (sp.plot.balloon_wireframe),
    (sp.plot.contour),
    (sp.plot.contour_map),
    (sp.plot.pcolor_map),
    (sp.plot.pcolor_sphere)])
def test_spherical_default(function):
    """Test all spherical plots with default arguments."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    filename = f'{function.__name__}_default'
    create_figure()
    function(coords, data)
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)

    # test error for invalid inputs
    match = 'data must be a 1D array with the same cshape as the coordinates'
    with pytest.raises(ValueError, match=match):
        function(coords, 'data')

    match = 'coordinates must be a coordinates object.'
    with pytest.raises(ValueError, match=match):
        function('coords', data)


@pytest.mark.parametrize('function', [
    (sp.plot.contour),
    (sp.plot.contour_map),
    (sp.plot.pcolor_map)])
@pytest.mark.parametrize('cmap', [plt.get_cmap('plasma'), 'plasma', None])
def test_spherical_cmap(function, cmap):
    """Test all spherical plots with custom cmap argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    if isinstance(cmap, mpl.colors.Colormap):
        # if a colormap object is passed, use its name for the filename
        cmap_str = 'ColormapObject'
    else:
        # otherwise use the string representation of the colormap
        cmap_str = str(cmap)
    filename = f'{function.__name__}_cmap_{cmap_str}'
    create_figure()
    function(coords, data, cmap=cmap)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = 'cmap'
    with pytest.raises(ValueError, match=match):
        function(coords, data, cmap=5)


@pytest.mark.parametrize('function', [
    (sp.plot.balloon),
    (sp.plot.balloon_wireframe),
    (sp.plot.contour),
    (sp.plot.contour_map),
    (sp.plot.pcolor_map),
    (sp.plot.pcolor_sphere)])
@pytest.mark.parametrize('colorbar', [True, False])
def test_spherical_colorbar(function, colorbar):
    """Test all spherical plots with custom colorbar argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    filename = f'{function.__name__}_colorbar_{colorbar}'
    create_figure()
    function(coords, data, colorbar=colorbar)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = 'colorbar must be a boolean.'
    with pytest.raises(ValueError, match=match):
        function(coords, data, colorbar='colorbar')


@pytest.mark.parametrize('function', [
    (sp.plot.balloon),
    (sp.plot.balloon_wireframe),
    (sp.plot.contour),
    (sp.plot.contour_map),
    (sp.plot.pcolor_map),
    (sp.plot.pcolor_sphere)])
@pytest.mark.parametrize('limits', [None, (-1.5, 1.5), [-1.5, 1.5]])
def test_spherical_limits(function, limits):
    """Test all spherical plots with custom limits argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    if limits is None:
        limits_str = 'None'
    else:
        limits_str = f'{type(limits).__name__}'
    filename = f'{function.__name__}_limits_{limits_str}'
    create_figure()
    function(coords, data, limits=limits)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = (
        "limits must be a tuple or list containing the minimum and "
        "maximum values for the colormap or None.")
    with pytest.raises(ValueError, match=match):
        function(coords, data, limits='limits')
    with pytest.raises(ValueError, match=match):
        function(coords, data, limits=[0, 1, 2])


@pytest.mark.parametrize('function', [
    (sp.plot.balloon),
    (sp.plot.balloon_wireframe),
    (sp.plot.pcolor_sphere)])
@pytest.mark.parametrize('phase', [True, False])
def test_spherical_phase(function, phase):
    """Test spherical plots with custom  phase argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
    if phase:
        # use complex data to test phase plotting
        data = np.exp(1j * data)

    # do plotting
    filename = f'{function.__name__}_phase_{phase}'
    create_figure()
    function(coords, data, phase=phase)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = 'phase must be a boolean.'
    with pytest.raises(ValueError, match=match):
        function(coords, data, phase=5)


@pytest.mark.parametrize('function', [
    (sp.plot.pcolor_map),
    (sp.plot.contour_map),
    ])
@pytest.mark.parametrize('projection', ['mollweide', 'hammer'])
def test_spherical_projection(function, projection):
    """Test spherical plots with custom projection argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    filename = f'{function.__name__}_projection_{projection}'
    create_figure()
    function(coords, data, projection=projection)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = "The projection of the axis needs to be 'projection'"
    with pytest.raises(ValueError, match=match):
        function(coords, data, projection='projection')


@pytest.mark.parametrize('function', [
    sp.plot.contour_map,
    # placeholder after level implementation in sp.plot.contour,
    ])
@pytest.mark.parametrize('levels', [
    (-0.5, .5),
    [-0.5, .5],
    np.array((-0.5, .5)),
    ])
def test_spherical_levels(function, levels):
    """Test contour plots with custom level argument."""
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    filename = f'{function.__name__}_levels_{type(levels).__name__}'
    create_figure()
    function(coords, data, levels=levels)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test error for invalid inputs
    match = 'levels'
    with pytest.raises(ValueError, match=match):
        function(coords, data, levels='levels')


@pytest.mark.parametrize(("function", "projection"), [
    (sp.plot.balloon, '3d'),
    (sp.plot.balloon_wireframe, '3d'),
    (sp.plot.pcolor_sphere, '3d'),
    (sp.plot.contour, 'rectilinear'),
    (sp.plot.contour_map, 'mollweide'),
    (sp.plot.pcolor_map, 'mollweide'),
    ])
def test_data_plots_projection_input_and_return(function, projection):
    """
    Test all spherical plots with ax argument and check if this is
    also returned.
    """
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
    # do plotting
    create_figure()
    ax = plt.axes(projection=projection)

    (ax_out, _) = function(coords, data, ax=ax)
    # check if the returned axis is a 3D axis
    assert ax_out.name == projection

    # test error for invalid inputs
    match = f"The projection of the axis needs to be '{projection}'"
    with pytest.raises(ValueError, match=match):
        function(coords, data, ax=plt.axes(projection='polar'))


@pytest.mark.parametrize('function', [
    (sp.plot.scatter),
    (sp.plot.voronoi_cells_sphere),
    ])
def test_coordinates_plots_projection_input_and_return(function):
    """Test scatter plots with ax argument and check if this is also returned.
    """
    coords = sp.samplings.equal_area(n_max=0, n_points=12)
    # do plotting
    create_figure()
    ax = plt.axes(projection='3d')

    ax_out = function(coords, ax=ax)

    # check if the returned axis is a 3D axis
    assert ax_out.name == '3d'

    # test error for invalid inputs
    match = "The projection of the axis needs to be '3d'"
    with pytest.raises(ValueError, match=match):
        function(coords, ax=plt.axes())
