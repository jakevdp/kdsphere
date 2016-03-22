import numpy as np
from scipy.spatial import cKDTree

from .utils import spherical_to_cartesian


class KDSphere(object):
    """KD Tree for Spherical Data, built on scipy's cKDTree

    Parameters
    ----------
    data : array_like, shape (N, 2)
        (lon, lat) pairs measured in radians
    **kwargs :
        Additional arguments are passed to cKDTree
    """
    def __init__(self, data, **kwargs):
        self.data = np.asarray(data)
        self.data3d = spherical_to_cartesian(self.data)
        self.kdtree_ = cKDTree(self.data3d, **kwargs)

    def query(self, data, k=1, eps=0, **kwargs):
        """Query for k-nearest neighbors

        Parameters
        ----------
        data : array_like, shape (N, 2)
            (lon, lat) pairs measured in radians
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

        Returns
        -------
        d : array_like, float, shape=(N, k)
            The distances to the nearest neighbors
        i : array_like, int, shape=(N, k)
            The indices of the neighbors
        """
        data_3d, r = spherical_to_cartesian(data, return_radius=True)
        dist_3d, ind = self.kdtree_.query(data_3d, k=k, eps=eps, **kwargs)
        dist_2d = 2 * np.arcsin(dist_3d * 0.5 / r)
        return dist_2d, ind
