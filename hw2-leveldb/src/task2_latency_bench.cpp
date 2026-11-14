#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "leveldb/db.h"
#include "leveldb/options.h"
#include "leveldb/write_batch.h"

using Clock = std::chrono::steady_clock;

static std::string MakeKey(uint64_t x) {
    char buf[32];
    snprintf(buf, sizeof(buf), "key%020lu", x);
    return std::string(buf);
}

static std::string MakeValue(size_t value_size) {
    return std::string(value_size, 'x');
}

static double Percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    size_t idx = static_cast<size_t>((p / 100.0) * (data.size() - 1));
    return data[idx];
}

static void WriteSummary(
    const std::string& path,
    std::vector<double> latencies,
    double elapsed_sec
) {
    std::sort(latencies.begin(), latencies.end());

    double sum = 0.0;
    for (double x : latencies) sum += x;

    std::ofstream out(path);
    out << "total_ops,elapsed_sec,throughput_ops_per_sec,avg_us,p50_us,p90_us,p99_us,p999_us,max_us\n";

    out << latencies.size() << ","
        << elapsed_sec << ","
        << latencies.size() / elapsed_sec << ","
        << sum / latencies.size() << ","
        << Percentile(latencies, 50.0) << ","
        << Percentile(latencies, 90.0) << ","
        << Percentile(latencies, 99.0) << ","
        << Percentile(latencies, 99.9) << ","
        << latencies.back() << "\n";
}

static void WriteWindowStats(
    std::ofstream& out,
    int window_id,
    std::vector<double>& window
) {
    if (window.empty()) return;

    std::sort(window.begin(), window.end());

    double sum = 0.0;
    for (double x : window) sum += x;

    out << window_id << ","
        << window.size() << ","
        << sum / window.size() << ","
        << Percentile(window, 50.0) << ","
        << Percentile(window, 90.0) << ","
        << Percentile(window, 99.0) << ","
        << Percentile(window, 99.9) << ","
        << window.back() << "\n";

    window.clear();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage:\n"
                  << argv[0]
                  << " <db_path> <num_ops> <value_size> <summary_csv> <timeseries_csv>\n";
        return 1;
    }

    std::string db_path = argv[1];
    uint64_t num_ops = std::stoull(argv[2]);
    size_t value_size = std::stoull(argv[3]);
    std::string summary_csv = argv[4];
    std::string timeseries_csv = argv[5];

    leveldb::Options options;
    options.create_if_missing = true;

    options.write_buffer_size = 4 * 1024 * 1024;

    leveldb::DB* db = nullptr;
    leveldb::Status status = leveldb::DB::Open(options, db_path, &db);

    if (!status.ok()) {
        std::cerr << "Failed to open DB: " << status.ToString() << "\n";
        return 1;
    }

    leveldb::WriteOptions write_options;
    write_options.sync = false;

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint64_t> dist(0, num_ops * 10);

    std::string value = MakeValue(value_size);

    std::vector<double> all_latencies;
    all_latencies.reserve(num_ops);

    std::vector<double> window_latencies;

    std::ofstream ts(timeseries_csv);
    ts << "window_id,ops,avg_us,p50_us,p90_us,p99_us,p999_us,max_us\n";

    const uint64_t window_ops = 100000;

    auto bench_start = Clock::now();

    for (uint64_t i = 1; i <= num_ops; ++i) {
        uint64_t k = dist(rng);
        std::string key = MakeKey(k);

        auto start = Clock::now();
        status = db->Put(write_options, key, value);
        auto end = Clock::now();

        if (!status.ok()) {
            std::cerr << "Put failed at op " << i << ": " << status.ToString() << "\n";
            delete db;
            return 1;
        }

        double us = std::chrono::duration<double, std::micro>(end - start).count();

        all_latencies.push_back(us);
        window_latencies.push_back(us);

        if (i % window_ops == 0) {
            int window_id = static_cast<int>(i / window_ops);
            WriteWindowStats(ts, window_id, window_latencies);

            std::cout << "finished " << i << " ops"
                      << ", latest latency = " << us << " us"
                      << std::endl;
        }
    }

    if (!window_latencies.empty()) {
        int window_id = static_cast<int>(num_ops / window_ops) + 1;
        WriteWindowStats(ts, window_id, window_latencies);
    }

    auto bench_end = Clock::now();
    double elapsed_sec = std::chrono::duration<double>(bench_end - bench_start).count();

    WriteSummary(summary_csv, all_latencies, elapsed_sec);

    delete db;

    std::cout << "Task 2 latency benchmark finished.\n";
    std::cout << "Summary CSV: " << summary_csv << "\n";
    std::cout << "Timeseries CSV: " << timeseries_csv << "\n";

    return 0;
}
