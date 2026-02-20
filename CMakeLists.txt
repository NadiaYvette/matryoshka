cmake_minimum_required(VERSION 3.16)
project(matryoshka C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# SSE2 is baseline on x86-64; explicitly enable for clarity.
add_compile_options(-msse2 -O3 -Wall -Wextra -Wpedantic)

# ── Library ────────────────────────────────────────────────────
add_library(matryoshka STATIC
    src/matryoshka.c
    src/leaf.c
    src/inode.c
    src/alloc.c
    src/hierarchy.c
    src/arena.c
)
target_include_directories(matryoshka PUBLIC include)

# ── Tests ──────────────────────────────────────────────────────
enable_testing()

add_executable(test_matryoshka tests/test_matryoshka.c)
target_link_libraries(test_matryoshka matryoshka)
add_test(NAME unit_tests COMMAND test_matryoshka)

# ── Benchmarks ─────────────────────────────────────────────────
add_executable(bench_matryoshka bench/bench_matryoshka.c)
target_link_libraries(bench_matryoshka matryoshka)

# ── Comparison Benchmark (C++) ─────────────────────────────────
enable_language(CXX)

set(BENCH_COMPARE_SOURCES bench/bench_compare.cpp)

# Detect vendored libart
set(ART_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/libart/art.c")
if(EXISTS "${ART_SOURCE}")
    list(APPEND BENCH_COMPARE_SOURCES "${ART_SOURCE}")
    set(HAS_ART TRUE)
endif()

add_executable(bench_compare ${BENCH_COMPARE_SOURCES})
target_link_libraries(bench_compare matryoshka)
target_include_directories(bench_compare PRIVATE include bench)
set_target_properties(bench_compare PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(bench_compare PRIVATE -O3 -msse2 -Wall -Wextra -Wno-pedantic)

# Detect TLX (system install)
find_path(TLX_INCLUDE_DIR tlx/container/btree_set.hpp)
if(TLX_INCLUDE_DIR)
    target_compile_definitions(bench_compare PRIVATE HAS_TLX)
    target_include_directories(bench_compare PRIVATE "${TLX_INCLUDE_DIR}")
endif()

# Detect vendored libart
if(HAS_ART)
    target_compile_definitions(bench_compare PRIVATE HAS_ART)
    target_include_directories(bench_compare PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/libart")
endif()

# Detect abseil
find_package(absl QUIET)
if(absl_FOUND)
    target_compile_definitions(bench_compare PRIVATE HAS_ABSEIL)
    target_link_libraries(bench_compare absl::btree)
endif()
