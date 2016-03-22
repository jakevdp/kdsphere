import numpy as np
from numpy.testing import assert_allclose
from nose import SkipTest

from kdsphere.utils import spherical_to_cartesian


def generate_lon_lat(N, rseed=42):
    rand = np.random.RandomState(rseed)
    lon = 2 * np.pi * rand.rand(N)
    lat = np.pi * (0.5 - rand.rand(N))
    return lon, lat


def test_spherical_to_cartesian_vs_astropy():
    try:
        from astropy.coordinates import SkyCoord
    except ImportError:
        raise SkipTest('astropy not available')

    lon, lat = generate_lon_lat(100)

    data_apy = SkyCoord(lon, lat, unit='rad')
    data3d_apy = np.array(data_apy.cartesian.xyz.T)

    data = np.vstack([lon, lat]).T
    data3d = spherical_to_cartesian(data)

    assert_allclose(data3d_apy, data3d)
