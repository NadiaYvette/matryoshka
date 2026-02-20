/*
 * bench_compare.cpp -- Comparative benchmark: matryoshka vs other trees.
 *
 * Usage:
 *   bench_compare --library <name> --workload <name> --size <N>
 *   bench_compare --all
 *
 * Outputs JSON lines to stdout (one per benchmark run).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "wrappers.h"
#include "workloads.h"

static const char *ALL_LIBRARIES[] = {
    "matryoshka",
    "matryoshka_fence",
    "matryoshka_eytz",
    "std_set",
#ifdef HAS_ABSEIL
    "abseil_btree",
#endif
#ifdef HAS_TLX
    "tlx_btree",
#endif
#ifdef HAS_ART
    "libart",
#endif
    nullptr
};

static const char *ALL_WORKLOADS[] = {
    "seq_insert", "rand_insert", "rand_delete",
    "mixed", "ycsb_a", "ycsb_b", "search_after_churn",
    nullptr
};

static const size_t ALL_SIZES[] = {
    65536, 262144, 1048576, 4194304, 16777216
};
static const int NUM_SIZES = 5;

static void dispatch_library(const std::string &lib,
                              const std::vector<std::string> &workloads,
                              const std::vector<size_t> &sizes)
{
    if (lib == "matryoshka") {
        run_workloads<WrapperMatryoshka>(workloads, sizes);
    } else if (lib == "matryoshka_fence") {
        run_workloads<WrapperMatryoshkaFence>(workloads, sizes);
    } else if (lib == "matryoshka_eytz") {
        run_workloads<WrapperMatryoshkaEytzinger>(workloads, sizes);
    } else if (lib == "std_set") {
        run_workloads<WrapperStdSet>(workloads, sizes);
    }
#ifdef HAS_ABSEIL
    else if (lib == "abseil_btree") {
        run_workloads<WrapperAbseil>(workloads, sizes);
    }
#endif
#ifdef HAS_TLX
    else if (lib == "tlx_btree") {
        run_workloads<WrapperTlx>(workloads, sizes);
    }
#endif
#ifdef HAS_ART
    else if (lib == "libart") {
        run_workloads<WrapperArt>(workloads, sizes);
    }
#endif
    else {
        fprintf(stderr, "Unknown library: %s\n", lib.c_str());
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --library <name> --workload <name> --size <N>\n"
        "       %s --all\n\n"
        "Libraries: matryoshka, matryoshka_fence, matryoshka_eytz, std_set"
#ifdef HAS_ABSEIL
        ", abseil_btree"
#endif
#ifdef HAS_TLX
        ", tlx_btree"
#endif
#ifdef HAS_ART
        ", libart"
#endif
        "\n"
        "Workloads: seq_insert, rand_insert, rand_delete, mixed,\n"
        "           ycsb_a, ycsb_b, search_after_churn\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    std::vector<std::string> libraries;
    std::vector<std::string> workloads;
    std::vector<size_t> sizes;
    bool run_all = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            run_all = true;
        } else if (strcmp(argv[i], "--library") == 0 && i + 1 < argc) {
            libraries.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--workload") == 0 && i + 1 < argc) {
            workloads.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            sizes.push_back((size_t)atol(argv[++i]));
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (run_all) {
        for (const char **p = ALL_LIBRARIES; *p; p++)
            libraries.push_back(*p);
        for (const char **p = ALL_WORKLOADS; *p; p++)
            workloads.push_back(*p);
        for (int i = 0; i < NUM_SIZES; i++)
            sizes.push_back(ALL_SIZES[i]);
    }

    if (libraries.empty() || workloads.empty() || sizes.empty()) {
        usage(argv[0]);
        return 1;
    }

    for (const auto &lib : libraries)
        dispatch_library(lib, workloads, sizes);

    return 0;
}
