// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#include "eckit/geo/Search.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <ostream>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

#include "eckit/geo/Exceptions.h"
#include "eckit/geo/Grid.h"
#include "eckit/geo/LibEcKitGeo.h"
#include "eckit/geo/Trace.h"
#include "eckit/geo/projection/LonLatToXYZ.h"
#include "eckit/geo/util/mutex.h"
#include "eckit/log/Log.h"
#include "eckit/utils/MD5.h"


namespace eckit::geo {


namespace {


using KDTree = search::KDTree;


std::string loader(const spec::Spec& spec) {
    return spec.get_string("search-loader",
                           spec.get_bool("caching", LibEcKitGeo::caching()) ? LibEcKitGeo::searchLoader() : "memory");
}


bool fast_build(const spec::Spec& spec) {
    return spec.get_bool("search-fast-build", LibEcKitGeo::searchFastBuild());
}


std::string points_uid(const std::vector<PointXYZ>& points, const std::vector<size_t>* indices = nullptr) {
    MD5 hash;
    hash.add(points.size());
    for (const auto& p : points) {
        hash.add(p.X());
        hash.add(p.Y());
        hash.add(p.Z());
    }

    if (indices != nullptr) {
        hash.add("indices");
        for (auto i : *indices) {
            hash.add(i);
        }
    }

    return hash.digest();
}


std::string indices_uid(const std::vector<size_t>& indices) {
    MD5 hash;
    hash.add(indices.size());
    for (auto i : indices) {
        hash.add(i);
    }
    return hash.digest();
}


/// Calls fun(begin, end) on contiguous ranges of [0, n), on threads, rethrowing the first exception
template <typename Fun>
void parallel_for(size_t n, size_t threads, const Fun& fun) {
    constexpr size_t min_per_thread = 256;  // smaller batches aren't worth a thread

    threads = std::min(threads, (n + min_per_thread - 1) / min_per_thread);
    if (threads <= 1) {
        fun(size_t{0}, n);
        return;
    }

    std::vector<std::exception_ptr> errors(threads);
    std::vector<std::thread> pool;
    pool.reserve(threads);

    for (size_t t = 0; t < threads; ++t) {
        const auto begin = n * t / threads;
        const auto end   = n * (t + 1) / threads;
        pool.emplace_back([&fun, &errors, t, begin, end] {
            try {
                fun(begin, end);
            }
            catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }

    for (auto& thread : pool) {
        thread.join();
    }

    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}


void sort(search::Neighbours& v) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.index < b.index);
    });
}


void check_radius(double radius) {
    if (!(radius >= 0.)) {
        throw exception::SearchError("Search: invalid radius " + std::to_string(radius), Here());
    }
}


}  // namespace


struct Search::Shared {
    std::unique_ptr<KDTree> kdtree;
    std::unique_ptr<const projection::LonLatToXYZ> to_xyz;

    // size of the numbering of the results (e.g. grid size)
    size_t numbering = 0;
};


template <typename Fill>
void Search::build(const std::string& uid, size_t size, const Spec& spec, const Fill& fill) {
    auto& kdtree = shared_->kdtree;
    kdtree.reset(geo::search::KDTreeFactory::build(loader(spec), uid, size));

    util::lock_guard<KDTree> lock(*kdtree);

    if (kdtree->ready()) {
        Log::debug() << "Search: loaded " << *kdtree << std::endl;
        return;
    }

    Trace trace("Search: build " + uid);

    std::vector<KDTree::Value> values;
    fill(values);
    ASSERT(values.size() == size);

    if (fast_build(spec)) {
        kdtree->build(values);
    }
    else {
        for (const auto& value : values) {
            kdtree->insert(value);
        }
    }

    kdtree->commit();
}


Search::Search(const Grid& grid, const Spec& spec) : shared_(std::make_shared<Shared>()) {
    configure(spec);

    // (lon, lat) queries are on the figure of the grid points, as Grid::to_xyz
    const auto& figure = grid.figure();
    shared_->to_xyz    = std::make_unique<projection::LonLatToXYZ>(figure.a(), figure.b());

    // the k-d tree depends on the grid points and the figure they're converted on
    const auto uid = MD5{grid.uid() + shared_->to_xyz->spec_str()}.digest();

    shared_->numbering = grid.size();

    build(uid, grid.size(), spec, [&grid](std::vector<KDTree::Value>& values) {
        const auto xyz = grid.to_xyz();
        ASSERT(xyz.size() == 3);

        values.reserve(xyz[0].size());
        for (size_t i = 0; i < xyz[0].size(); ++i) {
            values.emplace_back(PointXYZ{xyz[0][i], xyz[1][i], xyz[2][i]}, i);
        }
    });
}


Search::Search(const Grid& grid, const std::vector<size_t>& indices, const Spec& spec) :
    shared_(std::make_shared<Shared>()) {
    configure(spec);

    const auto size = grid.size();
    for (auto i : indices) {
        if (i >= size) {
            throw exception::SearchError(
                "Search: index " + std::to_string(i) + " out of range (grid size " + std::to_string(size) + ")",
                Here());
        }
    }

    const auto& figure = grid.figure();
    shared_->to_xyz    = std::make_unique<projection::LonLatToXYZ>(figure.a(), figure.b());
    shared_->numbering = size;

    const auto uid = MD5{grid.uid() + shared_->to_xyz->spec_str() + indices_uid(indices)}.digest();

    build(uid, indices.size(), spec, [&grid, &indices](std::vector<KDTree::Value>& values) {
        const auto xyz = grid.to_xyz();
        ASSERT(xyz.size() == 3);

        values.reserve(indices.size());
        for (auto i : indices) {
            values.emplace_back(PointXYZ{xyz[0][i], xyz[1][i], xyz[2][i]}, i);
        }
    });
}


Search::Search(const std::vector<PointXYZ>& points, const Spec& spec) : shared_(std::make_shared<Shared>()) {
    configure(spec);

    shared_->to_xyz    = std::make_unique<projection::LonLatToXYZ>(spec);
    shared_->numbering = points.size();

    build(points_uid(points), points.size(), spec, [&points](std::vector<KDTree::Value>& values) {
        values.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            values.emplace_back(points[i], i);
        }
    });
}


Search::Search(const std::vector<PointXYZ>& points, const std::vector<size_t>& indices, const Spec& spec) :
    shared_(std::make_shared<Shared>()) {
    configure(spec);

    if (indices.size() != points.size()) {
        throw exception::SearchError("Search: expected one index per point (points: " + std::to_string(points.size()) +
                                         ", indices: " + std::to_string(indices.size()) + ")",
                                     Here());
    }

    shared_->to_xyz    = std::make_unique<projection::LonLatToXYZ>(spec);
    shared_->numbering = indices.empty() ? 0 : *std::max_element(indices.begin(), indices.end()) + 1;

    build(points_uid(points, &indices), points.size(), spec, [&points, &indices](std::vector<KDTree::Value>& values) {
        values.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            values.emplace_back(points[i], indices[i]);
        }
    });
}


Search::Search(const Search& other, const Spec& spec) : shared_(other.shared_) {
    configure(spec);
}


Search::~Search() = default;


size_t Search::size() const {
    return shared_->kdtree->size();
}


size_t Search::missing_index() const {
    return shared_->numbering;
}


size_t Search::threads_available() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0) {
        if (auto n = CPU_COUNT(&set); n > 0) {
            return static_cast<size_t>(n);
        }
    }
#endif

    return std::max(1U, std::thread::hardware_concurrency());
}


Search::MemoryUsage Search::footprint() const {
    return shared_->kdtree->footprint();
}


const search::KDTree& Search::kdtree() const {
    return *shared_->kdtree;
}


Search::MemoryUsage Search::estimate(size_t size, const Spec& spec) {
    return geo::search::KDTreeFactory::estimate(loader(spec), size);
}


std::unique_ptr<spec::Custom> Search::spec_kdtree(const Spec& spec) {
    auto kdtree = std::make_unique<spec::Custom>();

    if (spec.has("search-loader")) {
        kdtree->set("search-loader", spec.get_string("search-loader"));
    }

    for (const std::string key : {"caching", "search-fast-build"}) {
        if (spec.has(key)) {
            kdtree->set(key, spec.get_bool(key));
        }
    }

    return kdtree;
}


bool Search::spec_has_behaviour(const Spec& spec) {
    return spec.has("search") || spec.has("search-k") || spec.has("search-radius");
}


void Search::configure(const Spec& spec) {
    static const std::map<std::string, Mode> MODES{
        {"nn", Mode::NN},
        {"knn", Mode::KNN},
        {"radius", Mode::RADIUS},
        {"knn_or_radius", Mode::KNN_OR_RADIUS},
        {"knn_and_radius", Mode::KNN_AND_RADIUS},
    };

    const auto name = spec.get_string("search", "nn");
    auto it         = MODES.find(name);
    if (it == MODES.end()) {
        throw exception::SearchError("Search: unknown search '" + name + "'", Here());
    }

    mode_ = it->second;

    if (mode_ == Mode::KNN || mode_ == Mode::KNN_OR_RADIUS || mode_ == Mode::KNN_AND_RADIUS) {
        if (!spec.has("search-k")) {
            throw exception::SearchError("Search: search '" + name + "' requires 'search-k'", Here());
        }
        k_ = spec.get_unsigned("search-k");
    }

    if (mode_ == Mode::RADIUS || mode_ == Mode::KNN_OR_RADIUS || mode_ == Mode::KNN_AND_RADIUS) {
        if (!spec.has("search-radius")) {
            throw exception::SearchError("Search: search '" + name + "' requires 'search-radius'", Here());
        }
        radius_ = spec.get_double("search-radius");
        check_radius(radius_);
    }
}


Search::Neighbours Search::search(const Point& p) const {
    switch (mode_) {
        case Mode::NN:
            return {search_nn(p)};
        case Mode::KNN:
            return search_knn(p, k_);
        case Mode::RADIUS:
            return search_radius(p, radius_);
        case Mode::KNN_OR_RADIUS:
            return search_knn_or_radius(p, k_, radius_);
        case Mode::KNN_AND_RADIUS:
            return search_knn_and_radius(p, k_, radius_);
    }

    NOTIMP;
}


Search::Neighbour Search::search_nn(const Point& p) const {
    return shared_->kdtree->nearest_neighbour(to_xyz(p));
}


Search::Neighbours Search::search_knn(const Point& p, size_t k) const {
    if (k == 0) {
        return {};
    }

    if (k == 1) {
        return {search_nn(p)};
    }

    auto result = shared_->kdtree->k_nearest_neighbours(to_xyz(p), k);

    sort(result);
    return result;
}


Search::Neighbours Search::search_radius(const Point& p, double radius) const {
    check_radius(radius);

    auto result = shared_->kdtree->find_in_sphere(to_xyz(p), radius);

    sort(result);
    return result;
}


Search::Neighbours Search::search_knn_or_radius(const Point& p, size_t k, double radius) const {
    auto result = search_radius(p, radius);
    auto knn    = search_knn(p, k);

    result.insert(result.end(), knn.begin(), knn.end());

    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.index < b.index; });
    result.erase(
        std::unique(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.index == b.index; }),
        result.end());

    sort(result);
    return result;
}


Search::Neighbours Search::search_knn_and_radius(const Point& p, size_t k, double radius) const {
    check_radius(radius);

    auto result = search_knn(p, k);
    result.erase(std::remove_if(result.begin(), result.end(), [radius](const auto& n) { return n.distance > radius; }),
                 result.end());

    return result;
}


void Search::search_nn(size_t n, const double* lon, const double* lat, size_t* index, double* distance,
                       size_t threads) const {
    parallel_for(n, threads == 0 ? threads_available() : threads, [&](size_t begin, size_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto nn = search_nn(PointLonLat{lon[i], lat[i]});
            index[i]      = nn.index;
            distance[i]   = nn.distance;
        }
    });
}


void Search::search_knn(size_t n, const double* lon, const double* lat, size_t k, size_t* index, double* distance,
                        size_t threads) const {
    const auto missing = missing_index();

    parallel_for(n, threads == 0 ? threads_available() : threads, [&](size_t begin, size_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto knn = search_knn(PointLonLat{lon[i], lat[i]}, k);
            ASSERT(knn.size() <= k);

            auto* idx  = index + i * k;
            auto* dist = distance + i * k;
            for (size_t j = 0; j < knn.size(); ++j) {
                idx[j]  = knn[j].index;
                dist[j] = knn[j].distance;
            }

            std::fill(idx + knn.size(), idx + k, missing);
            std::fill(dist + knn.size(), dist + k, std::numeric_limits<double>::infinity());
        }
    });
}


void Search::search_radius(size_t n, const double* lon, const double* lat, double radius, std::vector<size_t>& offsets,
                           std::vector<size_t>& index, std::vector<double>& distance, size_t threads) const {
    check_radius(radius);

    std::vector<Neighbours> results(n);
    parallel_for(n, threads == 0 ? threads_available() : threads, [&](size_t begin, size_t end) {
        for (auto i = begin; i < end; ++i) {
            results[i] = search_radius(PointLonLat{lon[i], lat[i]}, radius);
        }
    });

    offsets.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + results[i].size();
    }

    index.resize(offsets[n]);
    distance.resize(offsets[n]);
    for (size_t i = 0; i < n; ++i) {
        auto j = offsets[i];
        for (const auto& r : results[i]) {
            index[j]    = r.index;
            distance[j] = r.distance;
            ++j;
        }
        Neighbours{}.swap(results[i]);  // release as we go
    }
}


PointXYZ Search::to_xyz(const Point& p) const {
    if (std::holds_alternative<PointXYZ>(p)) {
        return std::get<PointXYZ>(p);
    }

    if (const auto* q = std::get_if<PointLonLat>(&p); q != nullptr) {
        return shared_->to_xyz->fwd(*q);
    }

    throw exception::SearchError("Search: unsupported point type (supported: PointXYZ, PointLonLat)", Here());
}


void Search::print(std::ostream& out) const {
    out << "Search[kdtree=" << *shared_->kdtree << "]";
}


}  // namespace eckit::geo
