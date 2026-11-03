// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "eckit/container/KDTree.h"
#include "eckit/geo/search/KDTree.h"


namespace eckit::geo::search {


class Memory final : public KDTree {
public:

    // -- Types

    /// As eckit::KDMemory, not collecting statistics
    struct Alloc : NoStatistics<KDMemory> {
        // as KDMemory::deleteNode, recursing with this type (of the nodes, see SPNode::left)
        template <class Node>
        void deleteNode(Ptr p, const Node* n) {
            if (auto* node = static_cast<Node*>(p); node != nullptr) {
                deleteNode(node->left(*this), n);
                deleteNode(node->right(*this), n);
                delete node;
            }
        }
    };

    using Container = KDTreeX<TT<Traits, Alloc>>;

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

    Alloc alloc_;

    // queries don't modify it once built (no statistics), but aren't const (eckit::SPTree)
    mutable Container tree_{alloc_};
    size_t count_ = 0;

    // -- Overridden methods

    void print(std::ostream&) const override;
};


}  // namespace eckit::geo::search
