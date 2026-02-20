/*
 * workloads.h -- Benchmark workload functions, templated on wrapper type.
 *
 * Each workload generates keys outside the timed section, then measures
 * the hot loop with clock_gettime(CLOCK_MONOTONIC).  A volatile sink
 * prevents dead-code elimination.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>

/* ── Timing ─────────────────────────────────────────────────── */

static inline double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── PRNG (xorshift64) ──────────────────────────────────────── */

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 42) : s(seed ? seed : 1) {}
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (uint32_t)(s & 0xFFFFFFFF);
    }
    int32_t next_in(int32_t lo, int32_t hi) {
        return lo + (int32_t)(next() % (uint32_t)(hi - lo));
    }
};

/* ── JSON output ────────────────────────────────────────────── */

static inline void emit_json(const char *library, const char *workload,
                              size_t n, size_t ops, double elapsed)
{
    double mops = (double)ops / elapsed / 1e6;
    double ns   = elapsed / (double)ops * 1e9;
    printf("{\"library\":\"%s\",\"workload\":\"%s\","
           "\"n\":%zu,\"ops\":%zu,"
           "\"elapsed_sec\":%.6f,\"mops\":%.4f,\"ns_per_op\":%.2f}\n",
           library, workload, n, ops, elapsed, mops, ns);
    fflush(stdout);
}

/* ── Workload helpers ───────────────────────────────────────── */

/* Generate a shuffled permutation of [0, n) scaled by 2+1 (odd). */
static inline std::vector<int32_t> make_shuffled_keys(size_t n, uint64_t seed)
{
    std::vector<int32_t> keys(n);
    std::iota(keys.begin(), keys.end(), 0);
    /* Fisher-Yates shuffle with our PRNG. */
    Rng rng(seed);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rng.next() % (i + 1);
        std::swap(keys[i], keys[j]);
    }
    /* Scale to odd values for predecessor-search interest. */
    for (auto &k : keys) k = k * 2 + 1;
    return keys;
}

/* Generate sorted keys 1, 3, 5, ..., 2n-1. */
static inline std::vector<int32_t> make_sorted_keys(size_t n)
{
    std::vector<int32_t> keys(n);
    for (size_t i = 0; i < n; i++) keys[i] = (int32_t)(i * 2 + 1);
    return keys;
}

/* ── Workloads ──────────────────────────────────────────────── */

/*
 * 1. Sequential insert: insert keys 1, 3, 5, ..., 2N-1 in order.
 */
template<typename W>
void workload_seq_insert(size_t n)
{
    auto keys = make_sorted_keys(n);
    W w;

    double t0 = now_sec();
    for (size_t i = 0; i < n; i++)
        w.insert(keys[i]);
    double elapsed = now_sec() - t0;

    emit_json(W::name(), "seq_insert", n, n, elapsed);
}

/*
 * 2. Random insert: insert N unique random keys.
 */
template<typename W>
void workload_rand_insert(size_t n)
{
    auto keys = make_shuffled_keys(n, 42);
    W w;

    double t0 = now_sec();
    for (size_t i = 0; i < n; i++)
        w.insert(keys[i]);
    double elapsed = now_sec() - t0;

    emit_json(W::name(), "rand_insert", n, n, elapsed);
}

/*
 * 3. Random delete: bulk-load N sorted keys, then delete all in random order.
 */
template<typename W>
void workload_rand_delete(size_t n)
{
    auto sorted = make_sorted_keys(n);
    auto shuffled = make_shuffled_keys(n, 99);

    W w;
    w.bulk_load(sorted.data(), n);

    double t0 = now_sec();
    for (size_t i = 0; i < n; i++)
        w.remove(shuffled[i]);
    double elapsed = now_sec() - t0;

    emit_json(W::name(), "rand_delete", n, n, elapsed);
}

/*
 * 4. Mixed insert/delete: insert N keys, then N operations of
 *    50% insert (new key) / 50% delete (existing key).
 */
template<typename W>
void workload_mixed(size_t n)
{
    auto keys = make_sorted_keys(n);
    W w;
    w.bulk_load(keys.data(), n);

    /* Build operation sequence: interleave inserts of new keys and
       deletes of existing keys. */
    Rng rng(77);
    int32_t next_new = (int32_t)(n * 2 + 1);
    std::vector<int32_t> existing(keys);

    /* Shuffle existing for random deletes. */
    for (size_t i = existing.size() - 1; i > 0; i--) {
        size_t j = rng.next() % (i + 1);
        std::swap(existing[i], existing[j]);
    }

    size_t del_idx = 0;
    size_t ops = n;
    volatile bool sink = false;

    double t0 = now_sec();
    for (size_t i = 0; i < ops; i++) {
        if (i % 2 == 0) {
            /* insert a new key */
            sink = w.insert(next_new);
            next_new += 2;
        } else {
            /* delete an existing key */
            if (del_idx < existing.size()) {
                sink = w.remove(existing[del_idx++]);
            }
        }
    }
    double elapsed = now_sec() - t0;
    (void)sink;

    emit_json(W::name(), "mixed", n, ops, elapsed);
}

/*
 * 5. YCSB-A (write-heavy): 95% insert / 5% search, N operations.
 */
template<typename W>
void workload_ycsb_a(size_t n)
{
    W w;
    Rng rng(55);
    int32_t next_key = 1;
    size_t ops = n;
    volatile bool sink = false;

    double t0 = now_sec();
    for (size_t i = 0; i < ops; i++) {
        if (rng.next() % 100 < 95) {
            w.insert(next_key);
            next_key += 2;
        } else {
            int32_t q = rng.next_in(0, next_key);
            sink = w.search(q);
        }
    }
    double elapsed = now_sec() - t0;
    (void)sink;

    emit_json(W::name(), "ycsb_a", n, ops, elapsed);
}

/*
 * 6. YCSB-B (delete-heavy): pre-load N keys, then 50% delete / 50% search.
 */
template<typename W>
void workload_ycsb_b(size_t n)
{
    auto sorted = make_sorted_keys(n);
    auto shuffled = make_shuffled_keys(n, 88);

    W w;
    w.bulk_load(sorted.data(), n);

    Rng rng(66);
    size_t del_idx = 0;
    size_t ops = n;
    volatile bool sink = false;

    double t0 = now_sec();
    for (size_t i = 0; i < ops; i++) {
        if (i % 2 == 0 && del_idx < shuffled.size()) {
            sink = w.remove(shuffled[del_idx++]);
        } else {
            int32_t q = rng.next_in(0, (int32_t)(n * 2));
            sink = w.search(q);
        }
    }
    double elapsed = now_sec() - t0;
    (void)sink;

    emit_json(W::name(), "ycsb_b", n, ops, elapsed);
}

/*
 * 7. Search after churn: insert N, do N/2 mixed ops, then 5M searches.
 */
template<typename W>
void workload_search_after_churn(size_t n)
{
    auto keys = make_sorted_keys(n);
    W w;
    w.bulk_load(keys.data(), n);

    /* Churn phase (untimed): N/2 mixed insert/delete. */
    Rng rng(33);
    int32_t next_new = (int32_t)(n * 2 + 1);
    size_t churn = n / 2;
    for (size_t i = 0; i < churn; i++) {
        if (i % 2 == 0) {
            w.insert(next_new);
            next_new += 2;
        } else {
            int32_t victim = rng.next_in(1, (int32_t)(n * 2));
            w.remove(victim);
        }
    }

    /* Generate random queries. */
    size_t nq = 5000000;
    std::vector<int32_t> queries(nq);
    for (size_t i = 0; i < nq; i++)
        queries[i] = rng.next_in(0, next_new);

    volatile bool sink = false;

    /* Warm up. */
    for (size_t i = 0; i < 100000 && i < nq; i++)
        sink = w.search(queries[i]);

    double t0 = now_sec();
    for (size_t i = 0; i < nq; i++)
        sink = w.search(queries[i]);
    double elapsed = now_sec() - t0;
    (void)sink;

    emit_json(W::name(), "search_after_churn", n, nq, elapsed);
}

/* ── Workload dispatch ──────────────────────────────────────── */

typedef void (*workload_fn_t)(size_t n);

struct WorkloadEntry {
    const char *name;
    workload_fn_t fn;
};

template<typename W>
void run_workloads(const std::vector<std::string> &workloads,
                   const std::vector<size_t> &sizes)
{
    for (size_t n : sizes) {
        for (const auto &wl : workloads) {
            if (wl == "seq_insert")          workload_seq_insert<W>(n);
            else if (wl == "rand_insert")    workload_rand_insert<W>(n);
            else if (wl == "rand_delete")    workload_rand_delete<W>(n);
            else if (wl == "mixed")          workload_mixed<W>(n);
            else if (wl == "ycsb_a")         workload_ycsb_a<W>(n);
            else if (wl == "ycsb_b")         workload_ycsb_b<W>(n);
            else if (wl == "search_after_churn") workload_search_after_churn<W>(n);
            else fprintf(stderr, "Unknown workload: %s\n", wl.c_str());
        }
    }
}
