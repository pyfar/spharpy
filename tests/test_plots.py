import os
import pytest
import matplotlib.pyplot as plt
import spharpy as sp
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import matplotlib as mpl

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
create_baseline = True

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


# testing ---------------------------------------------------------------------
@pytest.mark.parametrize('function', [
    (sp.plot.balloon),
    (sp.plot.balloon_wireframe),
    (sp.plot.contour),
    (sp.plot.contour_map),
    (sp.plot.pcolor_map),
    (sp.plot.pcolor_sphere)])
def test_spherical_default(function):
    """Test all spherical plots with default arguments."""
    print(f"Testing: {function.__name__}")
    coords = sp.samplings.equal_area(n_max=0, n_points=500)
    data = np.sin(coords.colatitude) * np.cos(coords.azimuth)

    # do plotting
    filename = f'default_{function.__name__}'
    create_figure()
    function(coords, data)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)
