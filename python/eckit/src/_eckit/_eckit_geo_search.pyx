# SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0


cimport eckit_geo_search
from cpython.pycapsule cimport PyCapsule_GetPointer
from cython.operator cimport dereference
from libc.string cimport memcpy
from libcpp.vector cimport vector


# Python-friendly spec keys
_KEYS = {
    "loader": "search-loader",
    "k": "search-k",
    "radius": "search-radius",
    "fast_build": "search-fast-build",
}


def _spec(spec: dict) -> str:
    from eckit.geo._eckit_geo import _spec_str

    return _spec_str({_KEYS.get(k, k.replace("_", "-")): v for k, v in spec.items()})


cdef const eckit_geo_search.Grid* _grid_ptr(grid) except NULL:
    # eckit.geo.Grid is defined in another extension module
    return <const eckit_geo_search.Grid*>PyCapsule_GetPointer(
        grid._capsule(), "eckit::geo::Grid"
    )


cdef vector[double] _point(point) except *:
    return [float(c) for c in point]


def _numpy():
    try:
        import numpy
    except ImportError as e:  # numpy is optional, except for batch queries
        raise ImportError("eckit.geo.Search: batch queries require numpy") from e
    return numpy


def _coordinates(*arrays):
    """Coordinates as contiguous float64 arrays of the same shape, and that shape."""
    np = _numpy()

    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]  # keeps 0-d (scalars)
    shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != shape:
            raise ValueError(
                f"eckit.geo.Search: coordinates of different shapes {shape} and {a.shape}"
            )
    return [np.ascontiguousarray(a.reshape(-1)) for a in arrays], shape


def _threads(threads) -> int:
    if threads is None:
        return 0  # as available to the process
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"eckit.geo.Search: invalid threads={threads}")
    return threads


cdef vector[size_t] _subset(mask, indices, size_t size) except *:
    """Indices of a subset of size points, by mask (boolean array) or indices."""
    np = _numpy()

    if mask is not None:
        if indices is not None:
            raise ValueError("eckit.geo.Search: mask and indices are exclusive")
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (size,):
            raise ValueError(
                f"eckit.geo.Search: mask must be a boolean array of shape ({size},)"
            )
        indices = np.flatnonzero(mask)

    array = np.asarray(indices)
    if array.ndim != 1 or not (array.size == 0 or np.issubdtype(array.dtype, np.integer)):
        raise TypeError("eckit.geo.Search: indices must be a 1-D array of integers")
    if array.size > 0 and (array.min() < 0 or array.max() >= size):
        raise IndexError(f"eckit.geo.Search: indices out of range [0, {size})")

    cdef const size_t[::1] view = np.ascontiguousarray(array, dtype=np.uintp)
    cdef vector[size_t] result = vector[size_t](view.shape[0])
    cdef Py_ssize_t i
    for i in range(view.shape[0]):
        result[i] = view[i]
    return result


_INTERNAL = object()  # construction without a grid (from_lonlat, from_xyz)


cdef class Search:
    """Search on the points of a grid, following k-d tree algorithms.

    Points are (lon, lat) or geocentric (x, y, z); results are (index, distance), by
    increasing distance (chord length, in metres on the Earth).

    Searches are shared by grid and spec (loader, k, radius, search, ..., see
    eckit::geo::Search) in a cache, within its capacity (see search_cache_statistics).

    A search can be on a subset of the grid points (mask, a boolean array of the grid
    size, or indices), with results by grid index. Searches on other points are made
    with from_lonlat and from_xyz.

    Batch queries (nn, knn, radius) take arrays of (lon, lat) and return numpy arrays;
    they run on threads, releasing the GIL. Queries are thread-safe.
    """
    cdef eckit_geo_search.SearchHandle* _handle

    def __cinit__(self, grid, *, mask=None, indices=None, **spec):
        self._handle = NULL
        if grid is _INTERNAL:
            return

        cdef const eckit_geo_search.Grid* ptr = _grid_ptr(grid)

        if mask is None and indices is None:
            self._handle = eckit_geo_search.search_get(dereference(ptr), _spec(spec))
            return

        self._handle = eckit_geo_search.search_grid_subset(
            dereference(ptr), _subset(mask, indices, grid.size()), _spec(spec)
        )

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    @staticmethod
    def from_lonlat(lon, lat, **spec) -> "Search":
        """Search on (lon, lat) points (arrays, in degrees), on the figure of the spec
        (e.g. figure="wgs84", or R=6371229, see eckit::geo::Figure; default the grids
        default).

        For a subset, search on lon[sel], lat[sel] and map results with sel[index].
        """
        (lon, lat), _ = _coordinates(lon, lat)
        return _from_points(lon, lat, None, spec)

    @staticmethod
    def from_xyz(x, y, z, **spec) -> "Search":
        """Search on geocentric (x, y, z) points (arrays, in metres); (lon, lat) queries
        are on the figure of the spec (as from_lonlat)."""
        (x, y, z), _ = _coordinates(x, y, z)
        return _from_points(x, y, z, spec)

    # -- Single queries, a point (lon, lat) or (x, y, z)

    def search(self, point) -> list:
        """Search with the configured behaviour (nearest neighbour by default)."""
        return eckit_geo_search.search_point(dereference(self._handle), _point(point))

    def search_nn(self, point) -> tuple:
        return eckit_geo_search.search_nn(dereference(self._handle), _point(point))

    def search_knn(self, point, k) -> list:
        return eckit_geo_search.search_knn(dereference(self._handle), _point(point), k)

    def search_radius(self, point, radius) -> list:
        return eckit_geo_search.search_radius(
            dereference(self._handle), _point(point), radius
        )

    def search_knn_or_radius(self, point, k, radius) -> list:
        """Union of the k nearest neighbours and the neighbours within radius."""
        return eckit_geo_search.search_knn_or_radius(
            dereference(self._handle), _point(point), k, radius
        )

    def search_knn_and_radius(self, point, k, radius) -> list:
        """Intersection of the k nearest neighbours and the neighbours within radius."""
        return eckit_geo_search.search_knn_and_radius(
            dereference(self._handle), _point(point), k, radius
        )

    # -- Batch queries, arrays of (lon, lat) (in degrees) of any (same) shape

    def nn(self, lon, lat, *, threads=None) -> tuple:
        """Nearest neighbour of each (lon, lat): (distance, index), numpy arrays of the
        shape of lon and lat (distances in metres, as chord lengths).

        threads: number of threads (default: as available to the process).
        """
        np = _numpy()
        (lon, lat), shape = _coordinates(lon, lat)

        cdef size_t n = lon.shape[0]
        cdef size_t nthreads = _threads(threads)

        distance = np.empty(n, dtype=np.float64)
        index = np.empty(n, dtype=np.intp)

        cdef const double[::1] lon_v = lon
        cdef const double[::1] lat_v = lat
        cdef double[::1] dist_v = distance
        cdef Py_ssize_t[::1] index_v = index
        cdef const eckit_geo_search.SearchHandle* h = self._handle

        if n > 0:
            with nogil:
                eckit_geo_search.batch_nn(
                    dereference(h), n, &lon_v[0], &lat_v[0],
                    <size_t*>&index_v[0], &dist_v[0], nthreads
                )

        return distance.reshape(shape), index.reshape(shape)

    def knn(self, lon, lat, k, *, threads=None) -> tuple:
        """k nearest neighbours of each (lon, lat), by increasing distance:
        (distance, index), numpy arrays of shape (*lon.shape, k) (distances in metres,
        as chord lengths).

        If fewer than k points are searched, missing neighbours have distance inf and
        index missing_index (as scipy.spatial.KDTree.query).

        threads: number of threads (default: as available to the process).
        """
        np = _numpy()
        (lon, lat), shape = _coordinates(lon, lat)

        k = int(k)
        if k < 1:
            raise ValueError(f"eckit.geo.Search: invalid k={k}")

        cdef size_t n = lon.shape[0]
        cdef size_t kk = k
        cdef size_t nthreads = _threads(threads)

        distance = np.empty((n, kk), dtype=np.float64)
        index = np.empty((n, kk), dtype=np.intp)

        cdef const double[::1] lon_v = lon
        cdef const double[::1] lat_v = lat
        cdef double[:, ::1] dist_v = distance
        cdef Py_ssize_t[:, ::1] index_v = index
        cdef const eckit_geo_search.SearchHandle* h = self._handle

        if n > 0:
            with nogil:
                eckit_geo_search.batch_knn(
                    dereference(h), n, &lon_v[0], &lat_v[0], kk,
                    <size_t*>&index_v[0, 0], &dist_v[0, 0], nthreads
                )

        return distance.reshape(shape + (k,)), index.reshape(shape + (k,))

    def radius(self, lon, lat, radius, *, threads=None) -> tuple:
        """Neighbours within radius (in metres, as a chord length) of each (lon, lat),
        by increasing distance: (distance, index, offsets), numpy arrays (CSR-style)
        where the neighbours of query i (of lon and lat, flattened) are
        distance[offsets[i]:offsets[i + 1]] and index[offsets[i]:offsets[i + 1]].

        threads: number of threads (default: as available to the process).
        """
        np = _numpy()
        (lon, lat), _ = _coordinates(lon, lat)

        cdef size_t n = lon.shape[0]
        cdef double r = float(radius)
        cdef size_t nthreads = _threads(threads)

        cdef vector[size_t] offsets_v
        cdef vector[size_t] index_v
        cdef vector[double] dist_v

        cdef const double[::1] lon_v = lon
        cdef const double[::1] lat_v = lat
        cdef const eckit_geo_search.SearchHandle* h = self._handle

        cdef const double* lon_p = NULL
        cdef const double* lat_p = NULL
        if n > 0:
            lon_p = &lon_v[0]
            lat_p = &lat_v[0]

        with nogil:
            eckit_geo_search.batch_radius(
                dereference(h), n, lon_p, lat_p, r, offsets_v, index_v, dist_v, nthreads
            )

        return (
            _array(dist_v.data(), dist_v.size(), np.float64),
            _array(index_v.data(), index_v.size(), np.intp),
            _array(offsets_v.data(), offsets_v.size(), np.intp),
        )

    @property
    def missing_index(self) -> int:
        """Index of missing neighbours (knn): the size of the numbering of the results
        (e.g. the grid size), so an invalid index."""
        return eckit_geo_search.search_missing_index(dereference(self._handle))

    @property
    def footprint(self) -> dict:
        """Memory used (bytes), in the process and shared (shared memory, page
        cache)."""
        usage = eckit_geo_search.search_footprint(dereference(self._handle))
        return dict(zip(("memory", "shared"), usage))

    def __len__(self) -> int:
        return eckit_geo_search.search_size(dereference(self._handle))

    def __repr__(self) -> str:
        return eckit_geo_search.search_str(dereference(self._handle))


cdef _array(const void* data, size_t size, dtype):
    """Copy of size items (of 8 bytes) as a numpy array of dtype."""
    np = _numpy()

    out = np.empty(size, dtype=dtype)
    assert out.itemsize == 8

    cdef unsigned char[::1] view = out.view(np.uint8)
    if size > 0:
        memcpy(&view[0], data, size * 8)
    return out


cdef Search _from_points(a, b, c, spec):
    """Search on points (lon, lat) if c is None, otherwise (x, y, z)."""
    cdef const double[::1] a_v = a
    cdef const double[::1] b_v = b
    cdef const double[::1] c_v
    cdef size_t n = a_v.shape[0]

    cdef const double* a_p = NULL
    cdef const double* b_p = NULL
    cdef const double* c_p = NULL
    if n > 0:
        a_p = &a_v[0]
        b_p = &b_v[0]

    cdef Search search = Search.__new__(Search, _INTERNAL)
    if c is None:
        search._handle = eckit_geo_search.search_points_lonlat(n, a_p, b_p, _spec(spec))
    else:
        c_v = c
        if n > 0:
            c_p = &c_v[0]
        search._handle = eckit_geo_search.search_points_xyz(n, a_p, b_p, c_p, _spec(spec))
    return search


def search_cache_statistics() -> dict:
    """Cache of searches: hits, misses, evictions, size, and usage (bytes, in the
    process and shared)."""
    keys = ("hits", "misses", "evictions", "size", "memory", "shared")
    return dict(zip(keys, eckit_geo_search.search_cache_statistics()))


def search_cache_clear() -> None:
    """Evict all searches (those in use remain valid)."""
    eckit_geo_search.search_cache_clear()
