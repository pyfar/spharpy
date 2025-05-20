from spharpy import SamplingSphere


def test_sampling_sphere_init():
    sampling = SamplingSphere()
    assert isinstance(sampling, SamplingSphere)


def test_sampling_sphere_init_value():
    sampling = SamplingSphere(1, 0, 0, 0)
    assert isinstance(sampling, SamplingSphere)


def sampling_cube():
    """Helper function returning a cube sampling"""
    x = [1, -1, 0, 0, 0, 0]
    y = [0, 0, 1, -1, 0, 0]
    z = [0, 0, 0, 0, 1, -1]

    return x, y, z


def test_getter_n_max():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    assert sampling.n_max == n_max


def test_setter_n_max():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, 0)

    sampling.n_max = n_max
    assert sampling._n_max == n_max


def test_setting_weights():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    weights = np.ones(6)*4*np.pi/6
    sampling.weights = weights

    np.testing.assert_array_equal(sampling._weights, weights)
    assert sampling._weights.shape == (6,)

def test_setting_weights_invalid():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    weights = np.ones(5)*4*np.pi/6

    # Check that setting weights with a different length raises an error
    message = "Weights must have same size as csize."
    with pytest.raises(ValueError, match=message):
        sampling.weights = weights

    message = r"The sum of the weights must be equal to 4\*pi."
    with pytest.raises(ValueError, match=message):
        sampling.weights = np.ones(6)/6

    # Check that setting weights with a different length raises an error
    message = "Weights must have same size as csize."
    with pytest.raises(ValueError, match=message):
        sampling._set_weights(weights)

    message = r"The sum of the weights must be equal to 4\*pi."
    with pytest.raises(ValueError, match=message):
        sampling._set_weights(np.ones(6)/6)
