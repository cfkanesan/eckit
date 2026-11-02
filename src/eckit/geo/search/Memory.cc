// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#include "eckit/geo/search/Memory.h"

#include <ostream>

#include "eckit/geo/Exceptions.h"


namespace eckit::geo::search {


static const KDTreeBuilder<Memory> BUILDER("memory");


Memory::Memory(const std::string& /*uid*/, size_t size) : KDTree(size) {}


void Memory::build(std::vector<Value>& values) {
    ASSERT(count_ == 0);
    ASSERT(values.size() <= size());

    // native value type, avoiding conversions while partitioning
    std::vector<Container::Value> v(values.begin(), values.end());
    tree_.build(v);
    count_ = v.size();
}


void Memory::insert(const Value& value) {
    ASSERT(count_ < size());
    tree_.insert(value);
    ++count_;
}


Neighbour Memory::nearest_neighbour(const Point& p) const {
    return to_neighbour(tree_.nearestNeighbour(p));
}


Neighbours Memory::k_nearest_neighbours(const Point& p, size_t k) const {
    return to_neighbours(tree_.kNearestNeighbours(p, k));
}


Neighbours Memory::find_in_sphere(const Point& p, double radius) const {
    return to_neighbours(tree_.findInSphere(p, radius));
}


void Memory::print(std::ostream& out) const {
    out << "Memory[size=" << size() << "]";
}


}  // namespace eckit::geo::search
