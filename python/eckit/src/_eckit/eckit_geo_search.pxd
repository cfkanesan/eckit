# SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0


from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector


cdef extern from "eckit/geo/Grid.h" namespace "eckit::geo":
    cdef cppclass Grid:
        pass


cdef extern from * namespace "eckit::geo::python":
    """
    #include <memory>
    #include <sstream>
    #include <string>
    #include <utility>
    #include <vector>

    #include "eckit/geo/Exceptions.h"
    #include "eckit/geo/Grid.h"
    #include "eckit/geo/Point.h"
    #include "eckit/geo/Search.h"
    #include "eckit/geo/cache/SearchCache.h"
    #include "eckit/geo/projection/LonLatToXYZ.h"
    #include "eckit/parser/YAMLParser.h"
    #include "eckit/spec/Custom.h"

    namespace eckit::geo::python {

    using neighbour_t  = std::pair<size_t, double>;
    using neighbours_t = std::vector<neighbour_t>;
    using point_t      = std::vector<double>;

    // a cached search, kept in use (not evicted) while referenced
    struct SearchHandle {
        std::shared_ptr<const Search> search;
    };

    using handle_t = SearchHandle;

    inline std::unique_ptr<spec::Custom> to_spec(const std::string& spec) {
        return std::unique_ptr<spec::Custom>(
            spec::Custom::make_from_value(YAMLParser::decodeString(spec)));
    }

    inline SearchHandle* search_get(const Grid& grid, const std::string& spec) {
        return new SearchHandle{cache::SearchCache::instance().get(grid, *to_spec(spec))};
    }

    // on a subset of the grid points (not cached in the process, the k-d tree is by spec)
    inline SearchHandle* search_grid_subset(
        const Grid& grid, const std::vector<size_t>& indices, const std::string& spec) {
        return new SearchHandle{std::make_shared<const Search>(grid, indices, *to_spec(spec))};
    }

    // on points (X, Y, Z), numbered by indices if given (has_indices)
    inline SearchHandle* search_points_xyz(
        size_t n, const double* x, const double* y, const double* z,
        const std::vector<size_t>& indices, bool has_indices, const std::string& spec) {
        std::vector<PointXYZ> points;
        points.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            points.emplace_back(x[i], y[i], z[i]);
        }

        const auto custom = to_spec(spec);
        return new SearchHandle{has_indices
                                    ? std::make_shared<const Search>(points, indices, *custom)
                                    : std::make_shared<const Search>(points, *custom)};
    }

    // on points (lon, lat), on the figure of the spec as (lon, lat) queries
    inline SearchHandle* search_points_lonlat(
        size_t n, const double* lon, const double* lat,
        const std::vector<size_t>& indices, bool has_indices, const std::string& spec) {
        const auto custom = to_spec(spec);
        const projection::LonLatToXYZ to_xyz(*custom);

        std::vector<PointXYZ> points;
        points.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            points.emplace_back(to_xyz.fwd(PointLonLat{lon[i], lat[i]}));
        }

        return new SearchHandle{has_indices
                                    ? std::make_shared<const Search>(points, indices, *custom)
                                    : std::make_shared<const Search>(points, *custom)};
    }

    inline Point to_point(const point_t& c) {
        if (c.size() == 2) {
            return PointLonLat{c[0], c[1]};
        }
        if (c.size() == 3) {
            return PointXYZ{c[0], c[1], c[2]};
        }
        throw exception::SearchError(
            "Search: expected point (lon, lat) or (x, y, z)", Here());
    }

    inline neighbour_t to_neighbour(const Search::Neighbour& n) {
        return {n.index, n.distance};
    }

    inline neighbours_t to_neighbours(const Search::Neighbours& v) {
        neighbours_t result;
        result.reserve(v.size());
        for (const auto& n : v) {
            result.emplace_back(to_neighbour(n));
        }
        return result;
    }

    inline neighbours_t search_point(const handle_t& h, const point_t& p) {
        return to_neighbours(h.search->search(to_point(p)));
    }

    inline neighbour_t search_nn(const handle_t& h, const point_t& p) {
        return to_neighbour(h.search->search_nn(to_point(p)));
    }

    inline neighbours_t search_knn(const handle_t& h, const point_t& p, size_t k) {
        return to_neighbours(h.search->search_knn(to_point(p), k));
    }

    inline neighbours_t search_radius(const handle_t& h, const point_t& p, double r) {
        return to_neighbours(h.search->search_radius(to_point(p), r));
    }

    inline neighbours_t search_knn_or_radius(
        const handle_t& h, const point_t& p, size_t k, double r) {
        return to_neighbours(h.search->search_knn_or_radius(to_point(p), k, r));
    }

    inline neighbours_t search_knn_and_radius(
        const handle_t& h, const point_t& p, size_t k, double r) {
        return to_neighbours(h.search->search_knn_and_radius(to_point(p), k, r));
    }

    // batch queries, on threads (0: as available), without the GIL

    inline void batch_nn(const handle_t& h, size_t n, const double* lon, const double* lat,
                         size_t* index, double* distance, size_t threads) {
        h.search->search_nn(n, lon, lat, index, distance, threads);
    }

    inline void batch_knn(const handle_t& h, size_t n, const double* lon, const double* lat,
                          size_t k, size_t* index, double* distance, size_t threads) {
        h.search->search_knn(n, lon, lat, k, index, distance, threads);
    }

    inline void batch_radius(const handle_t& h, size_t n, const double* lon, const double* lat,
                             double radius, std::vector<size_t>& offsets,
                             std::vector<size_t>& index, std::vector<double>& distance,
                             size_t threads) {
        h.search->search_radius(n, lon, lat, radius, offsets, index, distance, threads);
    }

    inline size_t search_missing_index(const SearchHandle& h) {
        return h.search->missing_index();
    }

    inline size_t search_size(const SearchHandle& h) {
        return h.search->size();
    }

    inline std::pair<size_t, size_t> search_footprint(const SearchHandle& h) {
        const auto f = h.search->footprint();
        return {f.memory(), f.shared()};
    }

    inline std::string search_str(const SearchHandle& h) {
        std::ostringstream str;
        str << *h.search;
        return str.str();
    }

    inline std::vector<size_t> search_cache_statistics() {
        const auto& cache = cache::SearchCache::instance();
        const auto stats  = cache.statistics();
        const auto usage  = cache.usage();
        return {stats.hits, stats.misses, stats.evictions, cache.size(),
                usage.memory(), usage.shared()};
    }

    inline void search_cache_clear() {
        cache::SearchCache::instance().clear();
    }

    }
    """
    cdef cppclass SearchHandle:
        pass

    SearchHandle* search_get(const Grid&, const string& spec) except +
    SearchHandle* search_grid_subset(
        const Grid&, const vector[size_t]& indices, const string& spec
    ) except +
    SearchHandle* search_points_xyz(
        size_t n, const double* x, const double* y, const double* z,
        const vector[size_t]& indices, bint has_indices, const string& spec
    ) except +
    SearchHandle* search_points_lonlat(
        size_t n, const double* lon, const double* lat,
        const vector[size_t]& indices, bint has_indices, const string& spec
    ) except +

    vector[pair[size_t, double]] search_point(
        const SearchHandle&, const vector[double]&
    ) except +
    pair[size_t, double] search_nn(const SearchHandle&, const vector[double]&) except +
    vector[pair[size_t, double]] search_knn(
        const SearchHandle&, const vector[double]&, size_t
    ) except +
    vector[pair[size_t, double]] search_radius(
        const SearchHandle&, const vector[double]&, double
    ) except +
    vector[pair[size_t, double]] search_knn_or_radius(
        const SearchHandle&, const vector[double]&, size_t, double
    ) except +
    vector[pair[size_t, double]] search_knn_and_radius(
        const SearchHandle&, const vector[double]&, size_t, double
    ) except +

    void batch_nn(
        const SearchHandle&, size_t n, const double* lon, const double* lat,
        size_t* index, double* distance, size_t threads
    ) except + nogil
    void batch_knn(
        const SearchHandle&, size_t n, const double* lon, const double* lat,
        size_t k, size_t* index, double* distance, size_t threads
    ) except + nogil
    void batch_radius(
        const SearchHandle&, size_t n, const double* lon, const double* lat,
        double radius, vector[size_t]& offsets, vector[size_t]& index,
        vector[double]& distance, size_t threads
    ) except + nogil

    size_t search_missing_index(const SearchHandle&) except +
    size_t search_size(const SearchHandle&) except +
    pair[size_t, size_t] search_footprint(const SearchHandle&) except +
    string search_str(const SearchHandle&) except +

    vector[size_t] search_cache_statistics() except +
    void search_cache_clear() except +
