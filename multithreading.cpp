#include <random>
#include "pcg/pcg_random.hpp"
#include <iostream>
#include <future>
#include <algorithm>
#include <iomanip>
#include <chrono>

const size_t SEED = 1337;

class ParallelGenerators {
    size_t nthreads_;
    std::vector<pcg32> generators_;
    std::uniform_real_distribution<double> unif_;

public:
    ParallelGenerators(size_t nthreads, size_t jump_ahead_step)
        : generators_(nthreads, pcg32(SEED)),
          unif_(-1., 1.) {
        
        for (size_t i = 0; i < nthreads; ++i) {
            generators_ [i].discard(i*jump_ahead_step);
        }
    }

    double get(size_t thread_id) {
        assert(thread_id < nthreads_);
        return unif_(generators_[thread_id]);
    }
};

bool in_circle(double x, double y) {
    return x*x + y*y <= 1.;
}

double calculate_single_threaded(size_t n_iter) {
    size_t N = 0;

    pcg32 rng(SEED);
    std::uniform_real_distribution<double> unif(-1., 1.);

    for (size_t i = 0; i < n_iter; ++i) {
        double x = unif(rng);
        double y = unif(rng);

        if (in_circle(x, y)) {
            ++N;
        }
    }

    return 4. * N/n_iter;
}

auto make_worker(size_t w_id, ParallelGenerators& gens) {
    return [w_id, &gens] (size_t n_iter) {
        size_t N = 0;
        
        for (size_t j = 0; j < n_iter; ++j) {
            double x = gens.get(w_id);
            double y = gens.get(w_id);

            if (in_circle(x, y)) {
                ++N;
            }
        }

        return N;
    };
}

double calculate_multi_threaded(size_t n_iter, size_t n_threads) {
    if (n_iter % n_threads != 0) {
        n_iter += n_threads - n_iter % n_threads;
        std::cout << "Warning: n_iter is not a multiple of n_threads. Setting n_iter to "
                  << n_iter << std::endl;
    }

    size_t n_iter_per_thread = n_iter / n_threads;

    ParallelGenerators gens(n_threads, 4*n_iter_per_thread);
    std::vector<std::future<size_t>> Ns;

    for (size_t i = 0; i < n_threads; ++i) {
        Ns.push_back(std::async(make_worker(i, gens), n_iter_per_thread));
    }

    size_t N_total = 0;

    for (size_t i = 0; i < n_threads; ++i) {
        N_total += Ns[i].get();
    }

    return 4. * N_total/n_iter;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: multithreading <n_iter> <n_threads>" << std::endl;
        return 0;
    }

    size_t n_iter = atoll(argv[1]);
    size_t n_threads = atoll(argv[2]);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sthreaded = calculate_single_threaded(n_iter);
    auto t2 = std::chrono::high_resolution_clock::now();
    double mthreaded = calculate_multi_threaded(n_iter, n_threads);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "Single thread: " << sthreaded << std::endl;
    std::cout << "Computed in " << std::chrono::duration<double, std::milli>(t2 - t1).count() << "ms\n" << std::endl;

    std::cout << "Multiple threads: " << mthreaded << std::endl;
    std::cout << "Computed in " << std::chrono::duration<double, std::milli>(t3 - t2).count() << "ms\n" << std::endl;

    std::cout << std::scientific << "--> Difference: " << sthreaded - mthreaded << std::endl;
    
    return 0;
}