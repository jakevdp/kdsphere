import numpy as np
from numpy.testing import assert_allclose
from nose import SkipTest

from kdsphere import KDSphere


def generate_lon_lat(N, rseed=42):
    rand = np.random.RandomState(rseed)
    lon = 2 * np.pi * rand.rand(N)
    lat = np.pi * (0.5 - rand.rand(N))
    return lon, lat


def test_kdsphere_vs_astropy():
    try:
        from astropy.coordinates import SkyCoord
    except ImportError:
        raise SkipTest('astropy not available')

    lon1, lat1 = generate_lon_lat(100, rseed=1)
    lon2, lat2 = generate_lon_lat(100, rseed=2)

    coord1 = SkyCoord(lon1, lat1, unit='rad')
    coord2 = SkyCoord(lon2, lat2, unit='rad')
    i_apy, d2d_apy, d3d_apy = coord2.match_to_catalog_3d(coord1)

    data1 = np.array([lon1, lat1]).T
    data2 = np.array([lon2, lat2]).T
    kd = KDSphere(data1)
    dist, ind = kd.query(data2, k=1)

    assert_allclose(ind.ravel(), i_apy)
    assert_allclose(dist.ravel(), d2d_apy.radian)
