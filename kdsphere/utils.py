import numpy as np


def spherical_to_cartesian(data, return_radius=False):
    """Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    data : array, shape (N, 2) or (N, 3)
    a collection of (lon, lat) or (lon, lat, time) coordinates.
    lon and lat should be in radians.
    If times are not specified, all are set to 1.0.

    Returns
    -------
    data3D : array, shape (N, 3)
    A 3D Cartesian view of the data. If time is provided, it is mapped to
    the radius.
    """
    data = np.asarray(data, dtype=float)

    # Data should be two-dimensional
    if data.ndim != 2:
        raise ValueError("data.shape = {0} should be "
                        "(N, 2) or (N, 3)".format(data.shape))

    # Data should have 2 or 3 columns
    if data.shape[1] == 2:
        lon, lat = data.T
        r = 1.0
    elif data.shape[1] == 3:
        lon, lat, r = data.T
    else:
        raise ValueError("data.shape = {0} should be "
                        "(N, 2) or (N, 3)".format(data.shape))

    data3d = np.array([r * np.cos(lat) * np.cos(lon),
                   r * np.cos(lat) * np.sin(lon),
                   r * np.sin(lat)]).T

    if return_radius:
        return data3d, r
    else:
        return data3d
