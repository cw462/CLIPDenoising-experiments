#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define LIMIT_BYTES ((size_t)1073741824)
#define MIN_AVAILABLE_KIB ((uint64_t)8388608)
#define PATTERNS 9
#define CHUNK_WORDS ((size_t)16384)
#define MAX_ERRORS 16
#define DURATION 59.0

struct failure { uint64_t offset, cycle; uintptr_t address; unsigned pattern; uint32_t expected, observed; };
struct coverage { uint64_t writes, reads, completed_fills, completed_verifications, errors; };
static const char *names[PATTERNS] = {"zero", "all_ones", "one", "reported_float_bits_3d30b0b1", "reported_bits_complement", "alternating_aaaaaaaa", "alternating_55555555", "address_mix", "address_mix_complement"};

static double seconds(void) {
    struct timespec current;
    if (clock_gettime(CLOCK_MONOTONIC, &current)) { perror("clock_gettime"); exit(4); }
    return (double)current.tv_sec + (double)current.tv_nsec * 1e-9;
}

static uint64_t available_kib(void) {
    FILE *stream = fopen("/proc/meminfo", "r");
    char line[256]; uint64_t value = 0;
    if (!stream) return 0;
    while (fgets(line, sizeof line, stream)) {
        if (sscanf(line, "MemAvailable: %" SCNu64 " kB", &value) == 1) break;
    }
    fclose(stream); return value;
}

static uint32_t expected_word(unsigned pattern, size_t index, uint64_t cycle) {
    uint32_t address_mix = (uint32_t)index * UINT32_C(0x9e3779b9)
                         ^ (uint32_t)(index >> 5) ^ UINT32_C(0xa5a55a5a)
                         ^ (uint32_t)cycle * UINT32_C(0x7f4a7c15);
    switch (pattern) {
        case 0: return UINT32_C(0x00000000);
        case 1: return UINT32_C(0xffffffff);
        case 2: return UINT32_C(0x00000001);
        case 3: return UINT32_C(0x3d30b0b1);
        case 4: return UINT32_C(0xc2cf4f4e);
        case 5: return UINT32_C(0xaaaaaaaa);
        case 6: return UINT32_C(0x55555555);
        case 7: return address_mix;
        default: return ~address_mix;
    }
}

int main(void) {
    const double started = seconds(), deadline = started + DURATION;
    const uint64_t available_before = available_kib();
    if (available_before <= MIN_AVAILABLE_KIB) {
        printf("{\"schema\":\"bounded_private_scalar_memory_test_v1\",\"status\":\"not_started\",\"reason\":\"MemAvailable must exceed 8 GiB\",\"available_kib\":%" PRIu64 "}\n", available_before);
        return 3;
    }
    volatile uint32_t *memory = malloc(LIMIT_BYTES);
    if (!memory) {
        printf("{\"schema\":\"bounded_private_scalar_memory_test_v1\",\"status\":\"allocation_failed\",\"errno\":%d}\n", errno);
        return 3;
    }
    const size_t words = LIMIT_BYTES / sizeof(uint32_t);
    struct failure failures[MAX_ERRORS];
    struct coverage covered[PATTERNS]; memset(covered, 0, sizeof covered);
    uint64_t errors = 0, completed_cycles = 0, cycle = 0;
    unsigned pattern = 0; size_t current_word = 0;
    const char *phase = "fill";
    int expired = 0;
    while (!expired) {
        for (pattern = 0; pattern < PATTERNS; ++pattern) {
            phase = "fill"; current_word = 0;
            for (size_t start = 0; start < words; start += CHUNK_WORDS) {
                if (seconds() >= deadline) { expired = 1; goto finished; }
                size_t end = start + CHUNK_WORDS; if (end > words) end = words;
                for (size_t index = start; index < end; ++index) {
                    memory[index] = expected_word(pattern, index, cycle);
                }
                covered[pattern].writes += end-start; current_word = end;
            }
            covered[pattern].completed_fills++;
            phase = "verify"; current_word = 0;
            for (size_t start = 0; start < words; start += CHUNK_WORDS) {
                if (seconds() >= deadline) { expired = 1; goto finished; }
                size_t end = start + CHUNK_WORDS; if (end > words) end = words;
                for (size_t index = start; index < end; ++index) {
                    const uint32_t observed = memory[index];
                    const uint32_t expected = expected_word(pattern, index, cycle);
                    if (observed != expected) {
                        if (errors < MAX_ERRORS) {
                            failures[errors] = (struct failure){.offset=index*sizeof(uint32_t), .cycle=cycle,
                                .address=(uintptr_t)(memory+index), .pattern=pattern, .expected=expected, .observed=observed};
                        }
                        errors++; covered[pattern].errors++;
                    }
                }
                covered[pattern].reads += end-start; current_word = end;
            }
            covered[pattern].completed_verifications++;
        }
        completed_cycles++; cycle++;
    }
finished:;
    const double ended = seconds();
    const uint64_t available_after = available_kib();
    uint64_t verified_words = 0, written_words = 0;
    for (unsigned p = 0; p < PATTERNS; ++p) { verified_words += covered[p].reads; written_words += covered[p].writes; }
    printf("{\n  \"schema\":\"bounded_private_scalar_memory_test_v1\",\n  \"status\":\"%s\",\n  \"pid\":%ld,\n  \"allocation_bytes\":%zu,\n  \"max_allocation_bytes\":%zu,\n  \"threads\":1,\n  \"volatile_scalar_uint32\":true,\n  \"requested_ceiling_seconds\":60,\n  \"duration_limit_seconds\":59,\n  \"elapsed_test_seconds\":%.9f,\n  \"available_kib_before\":%" PRIu64 ",\n  \"available_kib_after\":%" PRIu64 ",\n  \"completed_cycles\":%" PRIu64 ",\n  \"written_bytes\":%" PRIu64 ",\n  \"verified_bytes\":%" PRIu64 ",\n  \"error_count\":%" PRIu64 ",\n  \"recorded_errors\":%" PRIu64 ",\n  \"stop_phase\":\"%s\",\n  \"stop_pattern\":\"%s\",\n  \"stop_cycle\":%" PRIu64 ",\n  \"last_completed_word_offset\":%zu,\n  \"patterns\":[\n",
        errors ? "mismatch_detected" : "no_mismatch_observed", (long)getpid(), LIMIT_BYTES, LIMIT_BYTES,
        ended-started, available_before, available_after, completed_cycles, written_words*sizeof(uint32_t), verified_words*sizeof(uint32_t),
        errors, errors < MAX_ERRORS ? errors : (uint64_t)MAX_ERRORS, phase, names[pattern], cycle, current_word);
    for (unsigned p = 0; p < PATTERNS; ++p) {
        printf("    {\"name\":\"%s\",\"written_bytes\":%" PRIu64 ",\"verified_bytes\":%" PRIu64 ",\"completed_fills\":%" PRIu64 ",\"completed_verifications\":%" PRIu64 ",\"errors\":%" PRIu64 "}%s\n",
            names[p], covered[p].writes*sizeof(uint32_t), covered[p].reads*sizeof(uint32_t), covered[p].completed_fills, covered[p].completed_verifications, covered[p].errors, p+1<PATTERNS?",":"");
    }
    printf("  ],\n  \"errors\":[\n");
    for (uint64_t index = 0; index < errors && index < MAX_ERRORS; ++index) {
        struct failure *failure = &failures[index];
        printf("    {\"byte_offset\":%" PRIu64 ",\"virtual_address\":\"0x%" PRIxPTR "\",\"cycle\":%" PRIu64 ",\"pattern\":\"%s\",\"expected\":\"0x%08" PRIx32 "\",\"observed\":\"0x%08" PRIx32 "\",\"bit_xor\":\"0x%08" PRIx32 "\"}%s\n",
            failure->offset, failure->address, failure->cycle, names[failure->pattern], failure->expected, failure->observed,
            failure->expected ^ failure->observed, index+1<errors && index+1<MAX_ERRORS?",":"");
    }
    printf("  ],\n  \"limitations\":[\"Only this process's 1 GiB allocation was accessed; virtual addresses do not identify physical DIMMs.\",\"A short successful sample cannot exclude intermittent faults elsewhere or prove hardware/root cause healthy.\",\"No caches were flushed and physical memory placement was not controlled.\"]\n}\n");
    free((void *)memory);
    return errors ? 2 : 0;
}
