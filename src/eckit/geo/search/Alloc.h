// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cstddef>
#include <iosfwd>

#include "eckit/container/KDTree.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/geo/search/KDTree.h"


/// Allocators of k-d trees (see eckit::KDTreeX), not collecting statistics so that queries don't write (concurrent)
namespace eckit::geo::search::alloc {


/// Statistics interface of eckit::StatCollector, not collecting
struct NoStatistics {
    void statsCall() {}
    void statsVisitNode() {}
    void statsDepth(size_t) {}
    void statsNewCandidateOK() {}
    void statsNewCandidateMiss() {}
    void statsCrossOver() {}
    void statsReset() {}
    void statsPrint(std::ostream&, bool) const {}
};


/// As eckit::KDMemory (nodes on the heap)
struct Memory : NoStatistics {
    using Ptr = void*;

    Ptr root() const { return nullptr; }
    void root(Ptr) {}

    template <class Node>
    Ptr convert(Node* p) {
        return p;
    }

    template <class Node>
    Node* convert(Ptr p, const Node*) {
        return static_cast<Node*>(p);
    }

    template <class Node, typename A>
    Node* newNode1(const A& a, const Node*) {
        ++count_;
        return new Node(a);
    }

    template <class Node, typename A, typename B>
    Node* newNode2(const A& a, const B& b, const Node*) {
        ++count_;
        return new Node(a, b);
    }

    template <class Node, typename A, typename B, typename C>
    Node* newNode3(const A& a, const B& b, const C& c, const Node*) {
        ++count_;
        return new Node(a, b, c);
    }

    template <class Node>
    void deleteNode(Ptr p, const Node*) {
        if (auto* n = static_cast<Node*>(p); n != nullptr) {
            deleteNode(n->left(*this), n);
            deleteNode(n->right(*this), n);
            delete n;
            --count_;
        }
    }

    size_t nbItems() const { return count_; }

private:

    size_t count_ = 0;
};


/// Read-only, over memory holding k-d tree nodes of the eckit::KDMapped layout (cache file, mapped or loaded)
struct ReadOnly : NoStatistics {
    using Ptr = size_t;

    /// @param base address of node 0 (unused), so node p is at base[p]
    ReadOnly(char* base, size_t count) : base_(base), count_(count) {}

    Ptr root() const { return count_ > 0 ? 1 : 0; }
    void root(Ptr) { NOTIMP; }

    template <class Node>
    Ptr convert(Node* p) {
        return p != nullptr ? static_cast<Ptr>(p - reinterpret_cast<Node*>(base_)) : 0;
    }

    template <class Node>
    Node* convert(Ptr p, const Node*) {
        return p != 0 ? reinterpret_cast<Node*>(base_) + p : nullptr;
    }

    template <class Node, class A>
    Node* newNode1(const A&, const Node*) {
        NOTIMP;
    }

    template <class Node, class A, class B>
    Node* newNode2(const A&, const B&, const Node*) {
        NOTIMP;
    }

    template <class Node, class A, class B, class C>
    Node* newNode3(const A&, const B&, const C&, const Node*) {
        NOTIMP;
    }

    // nodes are not owned, so the k-d tree destruction doesn't access them
    template <class Node>
    void deleteNode(Ptr, Node*) {}

    size_t nbItems() const { return count_; }

private:

    char* base_;
    size_t count_;
};


using MemoryKDTree   = KDTreeX<TT<Traits, Memory>>;
using ReadOnlyKDTree = KDTreeX<TT<Traits, ReadOnly>>;


static_assert(sizeof(MemoryKDTree::Node) == sizeof(KDTreeMemory<Traits>::Node), "k-d tree node layouts differ");
static_assert(sizeof(ReadOnlyKDTree::Node) == sizeof(KDTreeMapped<Traits>::Node), "k-d tree node layouts differ");

// loaded content is aligned at least as allocated (operator new, shared memory pages)
static_assert(alignof(ReadOnlyKDTree::Node) <= alignof(std::max_align_t), "k-d tree node alignment");


/// Read-only k-d tree over nodes in memory, its root set (so queries don't write)
struct ReadOnlyView {
    ReadOnlyView(char* base, size_t count) : alloc(base, count), kdtree(alloc) { kdtree.root_ = alloc.root(); }

    ReadOnlyView(const ReadOnlyView&)            = delete;
    ReadOnlyView(ReadOnlyView&&)                 = delete;
    ReadOnlyView& operator=(const ReadOnlyView&) = delete;
    ReadOnlyView& operator=(ReadOnlyView&&)      = delete;

    ReadOnly alloc;

    // queries don't modify it (its root is set, statistics are not collected), but aren't const (eckit::SPTree)
    mutable ReadOnlyKDTree kdtree;
};


}  // namespace eckit::geo::search::alloc
