/*
 * wrappers.h -- Uniform wrapper classes for tree/map libraries.
 *
 * Each wrapper provides: insert, remove, search (predecessor),
 * contains, bulk_load, size, clear, name.  All inline for the
 * compiler to optimize the hot loop.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <set>
#include <algorithm>

/* ── matryoshka (C API) ─────────────────────────────────────── */

#include "matryoshka.h"

class WrapperMatryoshka {
    matryoshka_tree_t *tree_ = nullptr;
public:
    static const char *name() { return "matryoshka"; }
    static const char *label() { return "Matryoshka B+ tree"; }

    WrapperMatryoshka() { tree_ = matryoshka_create(); }
    ~WrapperMatryoshka() { if (tree_) matryoshka_destroy(tree_); }

    WrapperMatryoshka(const WrapperMatryoshka &) = delete;
    WrapperMatryoshka &operator=(const WrapperMatryoshka &) = delete;

    bool insert(int32_t key) { return matryoshka_insert(tree_, key); }
    bool remove(int32_t key) { return matryoshka_delete(tree_, key); }
    bool search(int32_t key) const {
        int32_t r;
        return matryoshka_search(tree_, key, &r);
    }
    bool contains(int32_t key) const {
        return matryoshka_contains(tree_, key);
    }
    void bulk_load(const int32_t *keys, size_t n) {
        if (tree_) matryoshka_destroy(tree_);
        tree_ = matryoshka_bulk_load(keys, n);
    }
    size_t size() const { return matryoshka_size(tree_); }
    void clear() {
        if (tree_) matryoshka_destroy(tree_);
        tree_ = matryoshka_create();
    }
};

/* ── std::set (red-black tree) ──────────────────────────────── */

class WrapperStdSet {
    std::set<int32_t> set_;
public:
    static const char *name() { return "std_set"; }
    static const char *label() { return "std::set (RB tree)"; }

    bool insert(int32_t key) { return set_.insert(key).second; }
    bool remove(int32_t key) { return set_.erase(key) > 0; }
    bool search(int32_t key) const {
        auto it = set_.upper_bound(key);
        return it != set_.begin();
    }
    bool contains(int32_t key) const { return set_.count(key) > 0; }
    void bulk_load(const int32_t *keys, size_t n) {
        set_.clear();
        set_.insert(keys, keys + n);
    }
    size_t size() const { return set_.size(); }
    void clear() { set_.clear(); }
};

/* ── Abseil btree_set ───────────────────────────────────────── */

#ifdef HAS_ABSEIL
#include "absl/container/btree_set.h"

class WrapperAbseil {
    absl::btree_set<int32_t> set_;
public:
    static const char *name() { return "abseil_btree"; }
    static const char *label() { return "Abseil btree_set"; }

    bool insert(int32_t key) { return set_.insert(key).second; }
    bool remove(int32_t key) { return set_.erase(key) > 0; }
    bool search(int32_t key) const {
        auto it = set_.upper_bound(key);
        return it != set_.begin();
    }
    bool contains(int32_t key) const { return set_.count(key) > 0; }
    void bulk_load(const int32_t *keys, size_t n) {
        set_.clear();
        set_.insert(keys, keys + n);
    }
    size_t size() const { return set_.size(); }
    void clear() { set_.clear(); }
};
#endif

/* ── TLX btree_set ──────────────────────────────────────────── */

#ifdef HAS_TLX
#include <tlx/container/btree_set.hpp>

class WrapperTlx {
    tlx::btree_set<int32_t> set_;
public:
    static const char *name() { return "tlx_btree"; }
    static const char *label() { return "TLX btree_set"; }

    bool insert(int32_t key) { return set_.insert(key).second; }
    bool remove(int32_t key) { return set_.erase(key) > 0; }
    bool search(int32_t key) const {
        auto it = set_.upper_bound(key);
        return it != set_.begin();
    }
    bool contains(int32_t key) const { return set_.count(key) > 0; }
    void bulk_load(const int32_t *keys, size_t n) {
        set_.clear();
        set_.insert(keys, keys + n);
    }
    size_t size() const { return set_.size(); }
    void clear() { set_.clear(); }
};
#endif

/* ── libart (Adaptive Radix Tree) ───────────────────────────── */

#ifdef HAS_ART
extern "C" {
#include "art.h"
}

class WrapperArt {
    art_tree tree_;

    /* Convert int32_t to 4-byte big-endian with flipped sign bit
       so lexicographic order matches numeric order. */
    static void to_key(int32_t val, unsigned char buf[4]) {
        uint32_t u = (uint32_t)val ^ 0x80000000u;
        buf[0] = (u >> 24) & 0xFF;
        buf[1] = (u >> 16) & 0xFF;
        buf[2] = (u >> 8)  & 0xFF;
        buf[3] =  u        & 0xFF;
    }
public:
    static const char *name() { return "libart"; }
    static const char *label() { return "libart (ART)"; }

    WrapperArt()  { art_tree_init(&tree_); }
    ~WrapperArt() { art_tree_destroy(&tree_); }

    WrapperArt(const WrapperArt &) = delete;
    WrapperArt &operator=(const WrapperArt &) = delete;

    bool insert(int32_t key) {
        unsigned char buf[4];
        to_key(key, buf);
        void *old = art_insert(&tree_, buf, 4, (void *)(uintptr_t)(key + 1));
        return old == nullptr;
    }
    bool remove(int32_t key) {
        unsigned char buf[4];
        to_key(key, buf);
        return art_delete(&tree_, buf, 4) != nullptr;
    }
    /* ART has no native predecessor; use point lookup. */
    bool search(int32_t key) const {
        unsigned char buf[4];
        to_key(key, buf);
        return art_search(&tree_, buf, 4) != nullptr;
    }
    bool contains(int32_t key) const {
        unsigned char buf[4];
        to_key(key, buf);
        return art_search(&tree_, buf, 4) != nullptr;
    }
    void bulk_load(const int32_t *keys, size_t n) {
        art_tree_destroy(&tree_);
        art_tree_init(&tree_);
        for (size_t i = 0; i < n; i++) insert(keys[i]);
    }
    size_t size() const { return art_size(const_cast<art_tree *>(&tree_)); }
    void clear() {
        art_tree_destroy(&tree_);
        art_tree_init(&tree_);
    }
};
#endif
