#ifndef ZIPF_H
#define ZIPF_H

#include <cstdint>
#include <random>
#include <vector>

class ZipfGenerator {
public:
    ZipfGenerator(uint64_t n, double theta, uint64_t seed = 12345);

    // Return an integer in [0, n - 1].
    uint64_t Next();

    uint64_t Size() const;
    double Theta() const;

private:
    uint64_t n_;
    double theta_;

    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_;

    std::vector<double> cdf_;

    void BuildCDF();
};

#endif
