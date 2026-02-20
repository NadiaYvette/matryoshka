cmake_minimum_required(VERSION 3.16)
project(matryoshka C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# SSE2 is baseline on x86-64; explicitly enable for clarity.
# Use -DMT_SIMD=avx2 or -DMT_SIMD=avx512 for wider SIMD.
set(MT_SIMD "sse2" CACHE STRING "SIMD level: sse2, avx2, avx512")
if(MT_SIMD STREQUAL "avx512")
    add_compile_options(-msse2 -mavx2 -mavx512f -mavx512bw -O3 -Wall -Wextra -Wpedantic)
elseif(MT_SIMD STREQUAL "avx2")
    add_compile_options(-msse2 -mavx2 -O3 -Wall -Wextra -Wpedantic)
else()
    add_compile_options(-msse2 -O3 -Wall -Wextra -Wpedantic)
endif()

# ── Library ────────────────────────────────────────────────────
add_library(matryoshka STATIC
    src/matryoshka.c
    src/leaf.c
    src/inode.c
    src/alloc.c
    src/hierarchy.c
    src/arena.c
    src/superpage.c
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

# ── Dependency detection: system install preferred, submodule fallback ──

# libart — system shared library, or compile submodule source directly
find_library(ART_LIBRARY art)
find_path(ART_INCLUDE_DIR art.h)
if(ART_LIBRARY AND ART_INCLUDE_DIR)
    set(HAS_ART TRUE)
    set(ART_USE_SYSTEM TRUE)
else()
    set(ART_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/libart/src/art.c")
    if(EXISTS "${ART_SOURCE}")
        list(APPEND BENCH_COMPARE_SOURCES "${ART_SOURCE}")
        set(HAS_ART TRUE)
        set(ART_USE_SYSTEM FALSE)
    endif()
endif()

# TLX — system headers, or submodule include path (header-only btree_set)
find_path(TLX_INCLUDE_DIR tlx/container/btree_set.hpp)
if(NOT TLX_INCLUDE_DIR)
    set(TLX_SUBMODULE "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/tlx")
    if(EXISTS "${TLX_SUBMODULE}/tlx/container/btree_set.hpp")
        set(TLX_INCLUDE_DIR "${TLX_SUBMODULE}")
    endif()
endif()
if(TLX_INCLUDE_DIR)
    set(HAS_TLX TRUE)
endif()

# Abseil — system package, or build submodule via add_subdirectory
find_package(absl QUIET)
if(absl_FOUND)
    set(HAS_ABSEIL TRUE)
    set(ABSEIL_USE_SYSTEM TRUE)
else()
    set(ABSEIL_SUBMODULE "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/abseil-cpp")
    if(EXISTS "${ABSEIL_SUBMODULE}/CMakeLists.txt")
        set(ABSL_PROPAGATE_CXX_STD ON)
        set(ABSL_BUILD_TESTING OFF)
        set(BUILD_TESTING OFF)
        add_subdirectory("${ABSEIL_SUBMODULE}" "${CMAKE_BINARY_DIR}/abseil-cpp" EXCLUDE_FROM_ALL)
        set(HAS_ABSEIL TRUE)
        set(ABSEIL_USE_SYSTEM FALSE)
    endif()
endif()

# ── bench_compare target ──────────────────────────────────────
add_executable(bench_compare ${BENCH_COMPARE_SOURCES})
target_link_libraries(bench_compare matryoshka)
target_include_directories(bench_compare PRIVATE include bench)
set_target_properties(bench_compare PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
if(MT_SIMD STREQUAL "avx512")
    target_compile_options(bench_compare PRIVATE -O3 -msse2 -mavx2 -mavx512f -mavx512bw -Wall -Wextra -Wno-pedantic)
elseif(MT_SIMD STREQUAL "avx2")
    target_compile_options(bench_compare PRIVATE -O3 -msse2 -mavx2 -Wall -Wextra -Wno-pedantic)
else()
    target_compile_options(bench_compare PRIVATE -O3 -msse2 -Wall -Wextra -Wno-pedantic)
endif()

# Link libart
if(HAS_ART)
    target_compile_definitions(bench_compare PRIVATE HAS_ART)
    if(ART_USE_SYSTEM)
        target_include_directories(bench_compare PRIVATE "${ART_INCLUDE_DIR}")
        target_link_libraries(bench_compare "${ART_LIBRARY}")
    else()
        target_include_directories(bench_compare PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/bench/vendor/libart/src")
    endif()
endif()

# Link TLX (header-only)
if(HAS_TLX)
    target_compile_definitions(bench_compare PRIVATE HAS_TLX)
    target_include_directories(bench_compare PRIVATE "${TLX_INCLUDE_DIR}")
endif()

# Link abseil
if(HAS_ABSEIL)
    target_compile_definitions(bench_compare PRIVATE HAS_ABSEIL)
    target_link_libraries(bench_compare absl::btree)
endif()
