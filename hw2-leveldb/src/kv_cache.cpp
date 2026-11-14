#include "kv_cache.h"

LRUCache::LRUCache(size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes),
      current_bytes_(0),
      hits_(0),
      misses_(0) {}

size_t LRUCache::EntrySize(const std::string& key, const std::string& value) const {
    return key.size() + value.size();
}

void LRUCache::Touch(ListIt it) {
    lru_list_.splice(lru_list_.begin(), lru_list_, it);
}

bool LRUCache::Get(const std::string& key, std::string& value) {
    auto it = map_.find(key);

    if (it == map_.end()) {
        ++misses_;
        return false;
    }

    Touch(it->second);
    value = it->second->value;
    ++hits_;
    return true;
}

void LRUCache::Put(const std::string& key, const std::string& value) {
    if (capacity_bytes_ == 0) {
        return;
    }

    size_t new_size = EntrySize(key, value);

    if (new_size > capacity_bytes_) {
        Remove(key);
        return;
    }

    auto it = map_.find(key);

    if (it != map_.end()) {
        current_bytes_ -= it->second->size_bytes;
        it->second->value = value;
        it->second->size_bytes = new_size;
        current_bytes_ += new_size;
        Touch(it->second);
    } else {
        lru_list_.push_front(Entry{key, value, new_size});
        map_[key] = lru_list_.begin();
        current_bytes_ += new_size;
    }

    EvictIfNeeded();
}

void LRUCache::Remove(const std::string& key) {
    auto it = map_.find(key);

    if (it == map_.end()) {
        return;
    }

    current_bytes_ -= it->second->size_bytes;
    lru_list_.erase(it->second);
    map_.erase(it);
}

void LRUCache::Clear() {
    lru_list_.clear();
    map_.clear();
    current_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
}

void LRUCache::EvictIfNeeded() {
    while (current_bytes_ > capacity_bytes_ && !lru_list_.empty()) {
        auto last = --lru_list_.end();
        current_bytes_ -= last->size_bytes;
        map_.erase(last->key);
        lru_list_.erase(last);
    }
}

size_t LRUCache::SizeBytes() const {
    return current_bytes_;
}

size_t LRUCache::CapacityBytes() const {
    return capacity_bytes_;
}

size_t LRUCache::ItemCount() const {
    return map_.size();
}

uint64_t LRUCache::Hits() const {
    return hits_;
}

uint64_t LRUCache::Misses() const {
    return misses_;
}

double LRUCache::HitRate() const {
    uint64_t total = hits_ + misses_;
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(hits_) / static_cast<double>(total);
}
