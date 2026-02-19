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
