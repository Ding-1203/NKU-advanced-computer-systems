#include "zipf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

ZipfGenerator::ZipfGenerator(uint64_t n, double theta, uint64_t seed)
    : n_(n),
      theta_(theta),
      rng_(seed),
      dist_(0.0, 1.0) {
    if (n_ == 0) {
        throw std::invalid_argument("ZipfGenerator: n must be greater than 0");
    }

    if (theta_ < 0.0) {
        throw std::invalid_argument("ZipfGenerator: theta must be non-negative");
    }

    BuildCDF();
}

void ZipfGenerator::BuildCDF() {
    cdf_.resize(n_);

    double sum = 0.0;

    for (uint64_t i = 1; i <= n_; ++i) {
        sum += 1.0 / std::pow(static_cast<double>(i), theta_);
    }

    double cumulative = 0.0;

    for (uint64_t i = 1; i <= n_; ++i) {
        cumulative += (1.0 / std::pow(static_cast<double>(i), theta_)) / sum;
        cdf_[i - 1] = cumulative;
    }

    cdf_[n_ - 1] = 1.0;
}

uint64_t ZipfGenerator::Next() {
    double u = dist_(rng_);

    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);

    if (it == cdf_.end()) {
        return n_ - 1;
    }

    return static_cast<uint64_t>(it - cdf_.begin());
}

uint64_t ZipfGenerator::Size() const {
    return n_;
}

double ZipfGenerator::Theta() const {
    return theta_;
}
