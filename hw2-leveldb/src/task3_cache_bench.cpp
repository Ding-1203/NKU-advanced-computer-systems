#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "leveldb/cache.h"
#include "leveldb/db.h"
#include "leveldb/options.h"

#include "kv_cache.h"
#include "zipf.h"

using Clock = std::chrono::steady_clock;

static uint64_t PermuteKey(uint64_t x, uint64_t n) {
    const uint64_t A = 11400714819323198485ull;
    return (x * A) % n;
}

static std::string MakeKey(uint64_t id) {
    char buf[32];
    snprintf(buf, sizeof(buf), "key%020lu", id);
    return std::string(buf);
}

static std::string MakeValue(size_t value_size, uint64_t seed) {
    std::string value(value_size, 'x');
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = static_cast<char>('a' + ((seed + i) % 26));
    }
    return value;
}

struct BenchResult {
    std::string mode;
    std::string workload;
    uint64_t total_ops = 0;
    double elapsed_sec = 0.0;
    double qps = 0.0;
    double avg_latency_us = 0.0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    double hit_rate = 0.0;
};

static void AppendResult(const std::string& csv_path, const BenchResult& r) {
    bool need_header = false;
    {
        std::ifstream in(csv_path);
        need_header = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(csv_path, std::ios::app);

    if (need_header) {
        out << "mode,workload,total_ops,elapsed_sec,qps,avg_latency_us,"
            << "cache_hits,cache_misses,hit_rate\n";
    }

    out << r.mode << ","
        << r.workload << ","
        << r.total_ops << ","
        << std::fixed << std::setprecision(6) << r.elapsed_sec << ","
        << r.qps << ","
        << r.avg_latency_us << ","
        << r.cache_hits << ","
        << r.cache_misses << ","
        << r.hit_rate << "\n";
}

static leveldb::DB* OpenDB(
    const std::string& db_path,
    bool create_if_missing,
    bool enable_block_cache,
    size_t cache_size_bytes
) {
    leveldb::Options options;
    options.create_if_missing = create_if_missing;
    options.error_if_exists = false;

    leveldb::Cache* block_cache = nullptr;
    if (enable_block_cache) {
        block_cache = leveldb::NewLRUCache(cache_size_bytes);
    } else {
        block_cache = leveldb::NewLRUCache(0);
    }

    options.block_cache = block_cache;

    leveldb::DB* db = nullptr;
    leveldb::Status s = leveldb::DB::Open(options, db_path, &db);

    if (!s.ok()) {
        std::cerr << "Failed to open DB: " << s.ToString() << std::endl;
        delete block_cache;
        return nullptr;
    }

    return db;
}

static bool BuildDB(
    const std::string& db_path,
    uint64_t num_keys,
    size_t value_size
) {
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = false;

    leveldb::DB* db = nullptr;
    leveldb::Status s = leveldb::DB::Open(options, db_path, &db);

    if (!s.ok()) {
        std::cerr << "Open DB failed: " << s.ToString() << std::endl;
        return false;
    }

    leveldb::WriteOptions write_options;
    write_options.sync = false;

    std::string value = MakeValue(value_size, 0);

    auto start = Clock::now();

    for (uint64_t i = 0; i < num_keys; ++i) {
        std::string key = MakeKey(i);
        s = db->Put(write_options, key, value);

        if (!s.ok()) {
            std::cerr << "Put failed: " << s.ToString() << std::endl;
            delete db;
            return false;
        }

        if ((i + 1) % 100000 == 0) {
            std::cout << "BuildDB finished " << (i + 1) << " keys" << std::endl;
        }
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "BuildDB finished. keys=" << num_keys
              << ", elapsed=" << elapsed << " sec"
              << std::endl;

    delete db;
    return true;
}

static BenchResult RunPointGet(
    leveldb::DB* db,
    const std::string& mode,
    uint64_t num_keys,
    uint64_t ops,
    bool use_kv_cache,
    size_t kv_cache_size,
    double zipf_theta
) {
    leveldb::ReadOptions read_options;
    read_options.fill_cache = !use_kv_cache;

    ZipfGenerator zipf(num_keys, zipf_theta, 12345);
    LRUCache kv_cache(kv_cache_size);

    uint64_t total_cache_hits_before = kv_cache.Hits();
    uint64_t total_cache_misses_before = kv_cache.Misses();

    auto bench_start = Clock::now();

    for (uint64_t i = 0; i < ops; ++i) {
        uint64_t id = PermuteKey(zipf.Next(), num_keys);
        std::string key = MakeKey(id);
        std::string value;

        if (use_kv_cache) {
            if (!kv_cache.Get(key, value)) {
                leveldb::Status s = db->Get(read_options, key, &value);
                if (s.ok()) {
                    kv_cache.Put(key, value);
                }
            }
        } else {
            leveldb::Status s = db->Get(read_options, key, &value);
            (void)s;
        }
    }

    auto bench_end = Clock::now();
    double elapsed = std::chrono::duration<double>(bench_end - bench_start).count();

    BenchResult r;
    r.mode = mode;
    r.workload = "point_get";
    r.total_ops = ops;
    r.elapsed_sec = elapsed;
    r.qps = ops / elapsed;
    r.avg_latency_us = elapsed * 1000000.0 / ops;

    if (use_kv_cache) {
        r.cache_hits = kv_cache.Hits() - total_cache_hits_before;
        r.cache_misses = kv_cache.Misses() - total_cache_misses_before;
        r.hit_rate = kv_cache.HitRate();
    }

    return r;
}

static BenchResult RunRangeScan(
    leveldb::DB* db,
    const std::string& mode,
    uint64_t num_keys,
    uint64_t scan_ops,
    uint64_t scan_length,
    bool use_kv_cache,
    size_t kv_cache_size,
    double zipf_theta
) {
    leveldb::ReadOptions read_options;
    read_options.fill_cache = !use_kv_cache;

    ZipfGenerator zipf(num_keys, zipf_theta, 23456);
    LRUCache kv_cache(kv_cache_size);

    auto bench_start = Clock::now();

    uint64_t scanned_items = 0;

    for (uint64_t i = 0; i < scan_ops; ++i) {
        uint64_t start_id = zipf.Next();

        if (start_id + scan_length >= num_keys) {
            start_id = num_keys > scan_length ? num_keys - scan_length : 0;
        }

        if (use_kv_cache) {
            for (uint64_t j = 0; j < scan_length; ++j) {
                std::string key = MakeKey(start_id + j);
                std::string value;

                if (!kv_cache.Get(key, value)) {
                    leveldb::Status s = db->Get(read_options, key, &value);
                    if (s.ok()) {
                        kv_cache.Put(key, value);
                    }
                }
                ++scanned_items;
            }
        } else {
            std::string start_key = MakeKey(start_id);

            leveldb::Iterator* it = db->NewIterator(read_options);
            it->Seek(start_key);

            uint64_t count = 0;
            while (it->Valid() && count < scan_length) {
                volatile size_t consume = it->key().size() + it->value().size();
                (void)consume;
                it->Next();
                ++count;
                ++scanned_items;
            }

            delete it;
        }
    }

    auto bench_end = Clock::now();
    double elapsed = std::chrono::duration<double>(bench_end - bench_start).count();

    BenchResult r;
    r.mode = mode;
    r.workload = "range_scan";
    r.total_ops = scan_ops;
    r.elapsed_sec = elapsed;
    r.qps = scan_ops / elapsed;
    r.avg_latency_us = elapsed * 1000000.0 / scan_ops;

    if (use_kv_cache) {
        r.cache_hits = kv_cache.Hits();
        r.cache_misses = kv_cache.Misses();
        r.hit_rate = kv_cache.HitRate();
    }

    return r;
}

static BenchResult RunMixed(
    leveldb::DB* db,
    const std::string& mode,
    uint64_t num_keys,
    uint64_t ops,
    bool use_kv_cache,
    size_t kv_cache_size,
    double zipf_theta
) {
    leveldb::ReadOptions read_options;
    read_options.fill_cache = !use_kv_cache;

    leveldb::WriteOptions write_options;
    write_options.sync = false;

    ZipfGenerator zipf(num_keys, zipf_theta, 34567);
    LRUCache kv_cache(kv_cache_size);

    std::mt19937_64 rng(98765);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<uint64_t> write_key_dist(0, num_keys - 1);

    auto bench_start = Clock::now();

    for (uint64_t i = 0; i < ops; ++i) {
        double p = prob(rng);

        if (p < 0.8) {
            uint64_t id = PermuteKey(zipf.Next(), num_keys);
            std::string key = MakeKey(id);
            std::string value;

            if (use_kv_cache) {
                if (!kv_cache.Get(key, value)) {
                    leveldb::Status s = db->Get(read_options, key, &value);
                    if (s.ok()) {
                        kv_cache.Put(key, value);
                    }
                }
            } else {
                leveldb::Status s = db->Get(read_options, key, &value);
                (void)s;
            }
        } else {
            uint64_t id = write_key_dist(rng);
            std::string key = MakeKey(id);
            std::string value = MakeValue(1024, i);

            leveldb::Status s = db->Put(write_options, key, value);
            if (!s.ok()) {
                std::cerr << "Mixed Put failed: " << s.ToString() << std::endl;
            }

            if (use_kv_cache) {
                kv_cache.Put(key, value);
            }
        }
    }

    auto bench_end = Clock::now();
    double elapsed = std::chrono::duration<double>(bench_end - bench_start).count();

    BenchResult r;
    r.mode = mode;
    r.workload = "mixed_80r20w";
    r.total_ops = ops;
    r.elapsed_sec = elapsed;
    r.qps = ops / elapsed;
    r.avg_latency_us = elapsed * 1000000.0 / ops;

    if (use_kv_cache) {
        r.cache_hits = kv_cache.Hits();
        r.cache_misses = kv_cache.Misses();
        r.hit_rate = kv_cache.HitRate();
    }

    return r;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage:\n"
                  << argv[0]
                  << " <db_path> <output_csv> <mode> <num_keys> <value_size>"
                  << " <point_ops> <scan_ops> <scan_length> <mixed_ops>\n\n"
                  << "mode: build_only | block_cache | kv_cache\n";
        return 1;
    }

    std::string db_path = argv[1];
    std::string output_csv = argv[2];
    std::string mode = argv[3];

    uint64_t num_keys = std::stoull(argv[4]);
    size_t value_size = std::stoull(argv[5]);
    uint64_t point_ops = std::stoull(argv[6]);
    uint64_t scan_ops = std::stoull(argv[7]);
    uint64_t scan_length = std::stoull(argv[8]);
    uint64_t mixed_ops = std::stoull(argv[9]);

    const size_t cache_size = 128ULL * 1024ULL * 1024ULL;
    const double zipf_theta = 1.2;

    bool build_only = (mode == "build_only");
    bool use_kv_cache = (mode == "kv_cache");
    bool enable_block_cache = (mode == "block_cache");

    if (!build_only && !use_kv_cache && !enable_block_cache) {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return 1;
    }

    std::cout << "Task 3 Cache Benchmark\n";
    std::cout << "Mode        : " << mode << "\n";
    std::cout << "DB path     : " << db_path << "\n";
    std::cout << "num_keys    : " << num_keys << "\n";
    std::cout << "value_size  : " << value_size << "\n";
    std::cout << "cache_size  : " << cache_size << "\n";
    std::cout << "zipf_theta  : " << zipf_theta << "\n";

    std::ifstream check(db_path + "/CURRENT");

    if (build_only) {
        if (check.good()) {
            std::cout << "Database already exists. Skip building." << std::endl;
            return 0;
        }

        std::cout << "Build-only mode. Building database..." << std::endl;
        if (!BuildDB(db_path, num_keys, value_size)) {
            return 1;
        }

        std::cout << "Build-only finished." << std::endl;
        return 0;
    }

    if (!check.good()) {
        std::cerr << "Database not found. Please run build_only first." << std::endl;
        return 1;
    }

    std::cout << "Database exists. Start benchmark only." << std::endl;

    leveldb::DB* db = OpenDB(
        db_path,
        false,
        enable_block_cache,
        cache_size
    );

    if (db == nullptr) {
        return 1;
    }

    BenchResult point = RunPointGet(
        db, mode, num_keys, point_ops,
        use_kv_cache, cache_size, zipf_theta
    );
    AppendResult(output_csv, point);
    std::cout << "Finished point_get: QPS=" << point.qps
              << ", hit_rate=" << point.hit_rate << std::endl;

    BenchResult scan = RunRangeScan(
        db, mode, num_keys, scan_ops, scan_length,
        use_kv_cache, cache_size, zipf_theta
    );
    AppendResult(output_csv, scan);
    std::cout << "Finished range_scan: QPS=" << scan.qps
              << ", hit_rate=" << scan.hit_rate << std::endl;

    BenchResult mixed = RunMixed(
        db, mode, num_keys, mixed_ops,
        use_kv_cache, cache_size, zipf_theta
    );
    AppendResult(output_csv, mixed);
    std::cout << "Finished mixed: QPS=" << mixed.qps
              << ", hit_rate=" << mixed.hit_rate << std::endl;

    delete db;

    return 0;
}
