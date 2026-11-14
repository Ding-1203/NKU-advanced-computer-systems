#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>

class LRUCache {
public:
    explicit LRUCache(size_t capacity_bytes);

    bool Get(const std::string& key, std::string& value);
    void Put(const std::string& key, const std::string& value);
    void Remove(const std::string& key);
    void Clear();

    size_t SizeBytes() const;
    size_t CapacityBytes() const;
    size_t ItemCount() const;

    uint64_t Hits() const;
    uint64_t Misses() const;
    double HitRate() const;

private:
    struct Entry {
        std::string key;
        std::string value;
        size_t size_bytes;
    };

    using ListIt = std::list<Entry>::iterator;

    size_t capacity_bytes_;
    size_t current_bytes_;

    std::list<Entry> lru_list_;
    std::unordered_map<std::string, ListIt> map_;

    uint64_t hits_;
    uint64_t misses_;

    size_t EntrySize(const std::string& key, const std::string& value) const;
    void Touch(ListIt it);
    void EvictIfNeeded();
};

#endif
