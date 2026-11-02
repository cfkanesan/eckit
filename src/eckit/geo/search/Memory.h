// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "eckit/geo/search/Alloc.h"
#include "eckit/geo/search/KDTree.h"


namespace eckit::geo::search {


class Memory final : public KDTree {
public:

    // -- Types

    using Container = alloc::MemoryKDTree;

    // -- Constructors

    Memory(const std::string& uid, size_t size);

    // -- Class methods

    static MemoryUsage estimate(size_t size) { return {size * sizeof(Container::Node), 0}; }

    // -- Overridden methods

    void build(std::vector<Value>&) override;
    void insert(const Value&) override;

    Neighbour nearest_neighbour(const Point&) const override;
    Neighbours k_nearest_neighbours(const Point&, size_t k) const override;
    Neighbours find_in_sphere(const Point&, double radius) const override;

    bool ready() override { return false; }
    void commit() override {}

    MemoryUsage footprint() const override { return {count_ * sizeof(Container::Node), 0}; }

private:

    // -- Members

    alloc::Memory alloc_;

    // queries don't modify it once built (its root is set, statistics are not collected), but aren't const
    mutable Container tree_{alloc_};

    size_t count_ = 0;

    // -- Overridden methods

    void print(std::ostream&) const override;
};


}  // namespace eckit::geo::search
