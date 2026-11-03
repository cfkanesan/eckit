# SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

# Batch searches (numpy arrays of (lon, lat) queries), searches on a subset of the grid
# points and on other points


from concurrent.futures import ThreadPoolExecutor

import pytest

from eckit.geo import Grid
from eckit.geo import Search

np = pytest.importorskip("numpy")


LOADERS = ("memory", "mmap", "file-io")


def random_lonlat(n, seed=0, shape=None):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    if shape is not None:
        lon, lat = lon.reshape(shape), lat.reshape(shape)
    return lon, lat


def brute_force_knn(x, y, z, qx, qy, qz, k):
    """k nearest (distance, index) by brute force, ties by index."""
    d = np.sqrt((x[None, :] - qx[:, None]) ** 2 + (y[None, :] - qy[:, None]) ** 2 + (z[None, :] - qz[:, None]) ** 2)
    order = np.lexsort((np.broadcast_to(np.arange(len(x)), d.shape), d), axis=-1)[:, :k]
    return np.take_along_axis(d, order, axis=-1), order


def to_xyz(lon, lat, R):
    lon, lat = np.radians(lon), np.radians(lat)
    return R * np.cos(lat) * np.cos(lon), R * np.cos(lat) * np.sin(lon), R * np.sin(lat)


@pytest.mark.parametrize("loader", LOADERS)
def test_knn_batch_is_single_queries(loader):
    grid = Grid(grid=[5, 5])
    search = grid.search(loader=loader)
    lon, lat = random_lonlat(500, seed=1)

    distance, index = search.knn(lon, lat, 4)
    assert distance.shape == index.shape == (500, 4)
    assert distance.dtype == np.float64
    assert index.dtype == np.intp

    for i in range(len(lon)):
        single = search.search_knn((lon[i], lat[i]), 4)
        assert list(index[i]) == [j for j, _ in single]
        assert list(distance[i]) == [d for _, d in single]

    # by increasing distance
    assert np.all(np.diff(distance, axis=1) >= 0)


@pytest.mark.parametrize("loader", LOADERS)
def test_nn_batch_is_single_queries(loader):
    grid = Grid(grid=[5, 5])
    search = grid.search(loader=loader)
    lon, lat = random_lonlat(500, seed=2)

    distance, index = search.nn(lon, lat)
    assert distance.shape == index.shape == (500,)

    for i in range(len(lon)):
        assert (index[i], distance[i]) == search.search_nn((lon[i], lat[i]))


@pytest.mark.parametrize("loader", LOADERS)
def test_radius_batch_is_single_queries(loader):
    grid = Grid(grid=[5, 5])
    search = grid.search(loader=loader)
    lon, lat = random_lonlat(300, seed=3)
    radius = 600e3

    distance, index, offsets = search.radius(lon, lat, radius)
    assert offsets.shape == (len(lon) + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(index) == len(distance)
    assert np.all(distance <= radius)

    for i in range(len(lon)):
        single = search.search_radius((lon[i], lat[i]), radius)
        assert list(index[offsets[i] : offsets[i + 1]]) == [j for j, _ in single]
        assert list(distance[offsets[i] : offsets[i + 1]]) == [d for _, d in single]


def test_knn_is_brute_force():
    grid = Grid(grid=[10, 10])
    search = grid.search(loader="memory")
    lon, lat = random_lonlat(200, seed=4)

    x, y, z = map(np.asarray, grid.to_xyz())
    R = grid.figure.R
    expected_distance, expected_index = brute_force_knn(x, y, z, *to_xyz(lon, lat, R), 6)

    distance, index = search.knn(lon, lat, 6)
    np.testing.assert_allclose(distance, expected_distance, rtol=0, atol=1e-6)

    # away from ties (the poles are represented by coincident points)
    away = np.abs(lat) < 80
    assert np.array_equal(index[away], expected_index[away])


def test_batch_shapes():
    search = Grid(grid=[5, 5]).search()

    lon, lat = random_lonlat(12, seed=5, shape=(3, 4))
    distance, index = search.knn(lon, lat, 2)
    assert distance.shape == index.shape == (3, 4, 2)

    distance, index = search.nn(lon, lat)
    assert distance.shape == index.shape == (3, 4)
    assert index[1, 2] == search.search_nn((lon[1, 2], lat[1, 2]))[0]

    # scalars, lists
    distance, index = search.nn(7.44, 46.95)
    assert distance.shape == index.shape == ()
    assert search.knn([7.44], [46.95], 3)[1].shape == (1, 3)

    # no queries
    assert search.knn([], [], 3)[0].shape == (0, 3)
    assert search.nn([], [])[0].shape == (0,)
    _, _, offsets = search.radius([], [], 1e3)
    assert list(offsets) == [0]

    with pytest.raises(ValueError):
        search.nn([1, 2], [3])
    with pytest.raises(ValueError):
        search.knn([1], [2], 0)


def test_threads():
    search = Grid(grid=[2, 2]).search()
    lon, lat = random_lonlat(5000, seed=6)

    reference = search.knn(lon, lat, 8, threads=1)
    for threads in (2, 7, None):
        result = search.knn(lon, lat, 8, threads=threads)
        assert np.array_equal(result[0], reference[0])
        assert np.array_equal(result[1], reference[1])

    with pytest.raises(ValueError):
        search.nn(lon, lat, threads=0)


def test_concurrent_python_threads():
    # queries are thread-safe, and batch queries release the GIL
    search = Grid(grid=[2, 2]).search()
    lon, lat = random_lonlat(5000, seed=7)
    reference = search.knn(lon, lat, 4, threads=1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: search.knn(lon, lat, 4, threads=2), range(8)))

    for distance, index in results:
        assert np.array_equal(distance, reference[0])
        assert np.array_equal(index, reference[1])


def test_knn_fewer_points_than_k():
    search = Search.from_lonlat([0, 90, 0], [0, 0, 90], loader="memory")
    assert search.missing_index == 3

    distance, index = search.knn([1, 89], [5, 10], 5)
    assert list(index[0, :3]) == [0, 2, 1] and list(index[1, :3]) == [1, 2, 0]
    assert np.all(index[:, 3:] == search.missing_index)
    assert np.all(np.isinf(distance[:, 3:]))
    assert np.all(np.isfinite(distance[:, :3]))


@pytest.mark.parametrize("loader", LOADERS)
def test_grid_subset_by_mask(loader):
    grid = Grid(grid=[5, 5])
    lat, lon = map(np.asarray, grid.to_latlons())

    # land-like mask: a band of latitudes
    mask = (lat > 20) & (lat < 60)
    search = grid.search(mask=mask, loader=loader)
    assert len(search) == mask.sum()
    assert search.missing_index == grid.size()

    qlon, qlat = random_lonlat(300, seed=8)
    distance, index = search.knn(qlon, qlat, 3)

    # results are by grid index, of masked points only
    assert np.all(mask[index])

    # as brute force on the masked points
    x, y, z = map(np.asarray, grid.to_xyz())
    selected = np.flatnonzero(mask)
    expected_distance, expected_index = brute_force_knn(
        x[selected], y[selected], z[selected], *to_xyz(qlon, qlat, grid.figure.R), 3
    )
    np.testing.assert_allclose(distance, expected_distance, rtol=0, atol=1e-6)
    assert np.array_equal(index, selected[expected_index])


def test_grid_subset_by_indices():
    grid = Grid(grid=[5, 5])
    indices = np.arange(0, grid.size(), 7)

    search = grid.search(indices=indices, loader="memory")
    assert len(search) == len(indices)

    _, index = search.nn(*random_lonlat(200, seed=9))
    assert np.all(np.isin(index, indices))

    # a grid point in the subset finds itself (away from the poles, of coincident points)
    lat, lon = grid.to_latlons()
    i = indices[100]
    assert abs(lat[i]) < 90
    distance, index = search.nn(lon[i], lat[i])
    assert index == i and distance < 1e-6

    # a mask is equivalent
    mask = np.zeros(grid.size(), dtype=bool)
    mask[indices] = True
    lon_q, lat_q = random_lonlat(100, seed=10)
    assert np.array_equal(
        grid.search(mask=mask, loader="memory").knn(lon_q, lat_q, 3)[1],
        search.knn(lon_q, lat_q, 3)[1],
    )


def test_grid_subset_errors():
    grid = Grid(grid=[30, 30])
    n = grid.size()

    with pytest.raises(ValueError):
        grid.search(mask=np.ones(n, dtype=bool), indices=[0])
    with pytest.raises(ValueError):
        grid.search(mask=np.ones(n + 1, dtype=bool))
    with pytest.raises(ValueError):
        grid.search(mask=np.ones(n, dtype=int))
    with pytest.raises(IndexError):
        grid.search(indices=[n])
    with pytest.raises(IndexError):
        grid.search(indices=[-1])
    with pytest.raises(TypeError):
        grid.search(indices=[0.5])


def test_from_lonlat_is_grid_search():
    grid = Grid(grid=[10, 10])
    lat, lon = grid.to_latlons()

    # same points, on the same (default) figure
    a = grid.search(loader="memory")
    b = Search.from_lonlat(lon, lat, loader="memory")
    assert len(a) == len(b) == grid.size()

    qlon, qlat = random_lonlat(200, seed=11)
    da, ia = a.knn(qlon, qlat, 4)
    db, ib = b.knn(qlon, qlat, 4)
    np.testing.assert_allclose(da, db, rtol=0, atol=1e-6)

    away = np.abs(qlat) < 80
    assert np.array_equal(ia[away], ib[away])


def test_from_lonlat_figure():
    lon, lat = [0, 90], [0, 0]

    # chord between two points on the equator, 90 degrees apart
    for spec, R in ((dict(R=1.0), 1.0), (dict(figure="wgs84"), 6378137.0)):
        distance, index = Search.from_lonlat(lon, lat, loader="memory", **spec).knn(0, 0, 2)
        assert list(index) == [0, 1]
        assert distance[1] == pytest.approx(R * np.sqrt(2))


def test_from_xyz():
    grid = Grid(grid=[10, 10])
    x, y, z = grid.to_xyz()

    a = grid.search(loader="memory")
    b = Search.from_xyz(x, y, z, loader="memory")

    qlon, qlat = random_lonlat(200, seed=14)
    da, ia = a.knn(qlon, qlat, 4)
    db, ib = b.knn(qlon, qlat, 4)
    np.testing.assert_allclose(da, db, rtol=0, atol=1e-6)

    # (x, y, z) single queries
    i, d = b.search_nn((x[42], y[42], z[42]))
    assert (i, d) == (42, 0.0)

    with pytest.raises(ValueError):
        Search.from_xyz(x, y, z[:-1])
