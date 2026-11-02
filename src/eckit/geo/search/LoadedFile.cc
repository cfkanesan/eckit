// SPDX-FileCopyrightText: 1996- European Centre for Medium-Range Weather Forecasts (ECMWF)
// SPDX-License-Identifier: Apache-2.0


#include "eckit/geo/search/LoadedFile.h"

#include <cstddef>
#include <cstring>

#include "eckit/container/KDMapped.h"
#include "eckit/container/KDTree.h"
#include "eckit/exception/Exceptions.h"


namespace eckit::geo::search {


namespace {


KDMappedHeader header(const char* content) {
    KDMappedHeader h(0, 0, 0);
    std::memcpy(&h, content, sizeof(h));
    return h;
}


// as KDMapped: items follow the header and metadata, aligned to the item size
char* items(char* content) {
    const auto h = header(content);
    return content + ((h.headerSize_ + h.metadataSize_ + h.itemSize_ - 1) / h.itemSize_) * h.itemSize_;
}


}  // namespace


struct LoadedFile::Content : alloc::ReadOnlyView {
    explicit Content(char* content) : ReadOnlyView(items(content), header(content).itemCount_) {}
};


LoadedFile::LoadedFile(const std::string& uid, size_t size) : MappedFile(uid, size) {}


LoadedFile::~LoadedFile() = default;


Neighbour LoadedFile::nearest_neighbour(const Point& p) const {
    return content_ ? to_neighbour(content_->kdtree.nearestNeighbour(p)) : MappedFile::nearest_neighbour(p);
}


Neighbours LoadedFile::k_nearest_neighbours(const Point& p, size_t k) const {
    return content_ ? to_neighbours(content_->kdtree.kNearestNeighbours(p, k)) : MappedFile::k_nearest_neighbours(p, k);
}


Neighbours LoadedFile::find_in_sphere(const Point& p, double radius) const {
    return content_ ? to_neighbours(content_->kdtree.findInSphere(p, radius)) : MappedFile::find_in_sphere(p, radius);
}


bool LoadedFile::ready() {
    if (content_) {
        return true;
    }

    if (!MappedFile::ready()) {
        return false;
    }

    load_content();
    return true;
}


void LoadedFile::commit() {
    MappedFile::commit();
    load_content();
}


void LoadedFile::load_content() {
    ASSERT(!content_);

    if (auto* content = load(); content != nullptr) {
        content_ = std::make_unique<Content>(content);
        close();  // the cache file mapping is no longer required
    }
}


}  // namespace eckit::geo::search
