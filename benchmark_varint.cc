#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>
#include <immintrin.h>

template <class T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

// Baseline Google Protobuf DecodeVarint64KnownSize
template <size_t N>
inline const uint8_t* BaselineDecodeVarint64KnownSize(const uint8_t* buffer, uint64_t* value) {
  uint64_t result = static_cast<uint64_t>(buffer[N - 1]) << (7 * (N - 1));
  for (size_t i = 0, offset = 0; i < N - 1; i++, offset += 7) {
    result += static_cast<uint64_t>(buffer[i] - 0x80) << offset;
  }
  *value = result;
  return buffer + N;
}

inline std::pair<bool, const uint8_t*> BaselineReadVarint64FromArray(
    const uint8_t* buffer, uint64_t* value) {
  const uint8_t* next;
  if (buffer[1] < 128) {
    next = BaselineDecodeVarint64KnownSize<2>(buffer, value);
  } else if (buffer[2] < 128) {
    next = BaselineDecodeVarint64KnownSize<3>(buffer, value);
  } else if (buffer[3] < 128) {
    next = BaselineDecodeVarint64KnownSize<4>(buffer, value);
  } else if (buffer[4] < 128) {
    next = BaselineDecodeVarint64KnownSize<5>(buffer, value);
  } else if (buffer[5] < 128) {
    next = BaselineDecodeVarint64KnownSize<6>(buffer, value);
  } else if (buffer[6] < 128) {
    next = BaselineDecodeVarint64KnownSize<7>(buffer, value);
  } else if (buffer[7] < 128) {
    next = BaselineDecodeVarint64KnownSize<8>(buffer, value);
  } else if (buffer[8] < 128) {
    next = BaselineDecodeVarint64KnownSize<9>(buffer, value);
  } else if (buffer[9] < 128) {
    next = BaselineDecodeVarint64KnownSize<10>(buffer, value);
  } else {
    return std::make_pair(false, buffer + 11);
  }
  return std::make_pair(true, next);
}

inline std::pair<bool, const uint8_t*> BaselineReadVarint32FromArray(
    uint32_t first_byte, const uint8_t* buffer, uint32_t* value) {
  const uint8_t* ptr = buffer;
  uint32_t b;
  uint32_t result = first_byte - 0x80;
  ++ptr;
  b = *(ptr++);
  result += b << 7;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 7;
  b = *(ptr++);
  result += b << 14;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 14;
  b = *(ptr++);
  result += b << 21;
  if (!(b & 0x80)) goto done;
  result -= 0x80 << 21;
  b = *(ptr++);
  result += b << 28;
  if (!(b & 0x80)) goto done;

  for (int i = 0; i < 10 - 5; i++) {
    b = *(ptr++);
    if (!(b & 0x80)) goto done;
  }
  return std::make_pair(false, ptr);

done:
  *value = result;
  return std::make_pair(true, ptr);
}

// ================= Implementation 1: BMI2 PEXT =================
inline std::pair<bool, const uint8_t*> PextReadVarint64FromArray(
    const uint8_t* buffer, uint64_t* value) {
  uint64_t first8;
  std::memcpy(&first8, buffer, sizeof(first8));

  uint64_t term_mask = (~first8) & 0x8080808080808080ULL;
  if (__builtin_expect(term_mask != 0, 1)) {
    int ctz = __builtin_ctzll(term_mask);
    int len = (ctz >> 3) + 1;
    uint64_t payload = _pext_u64(first8, 0x7f7f7f7f7f7f7f7fULL);
    *value = _bzhi_u64(payload, 7 * len);
    return {true, buffer + len};
  }

  if (buffer[8] < 128) {
    return {true, BaselineDecodeVarint64KnownSize<9>(buffer, value)};
  } else if (buffer[9] < 128) {
    return {true, BaselineDecodeVarint64KnownSize<10>(buffer, value)};
  }
  return {false, buffer + 11};
}

inline std::pair<bool, const uint8_t*> PextReadVarint32FromArray(
    uint32_t first_byte, const uint8_t* buffer, uint32_t* value) {
  uint64_t first8;
  std::memcpy(&first8, buffer, sizeof(first8));

  uint64_t term_mask = (~first8) & 0x8080808080808080ULL;
  if (__builtin_expect(term_mask != 0, 1)) {
    int ctz = __builtin_ctzll(term_mask);
    int len = (ctz >> 3) + 1;
    uint64_t payload = _pext_u64(first8, 0x7f7f7f7f7f7f7f7fULL);
    *value = static_cast<uint32_t>(_bzhi_u64(payload, 7 * len));
    return {true, buffer + len};
  }

  uint64_t val64;
  auto res = BaselineReadVarint64FromArray(buffer, &val64);
  if (res.first) {
    *value = static_cast<uint32_t>(val64);
  }
  return res;
}

// ================= Implementation 2: Portable Tree Parallel =================
inline std::pair<bool, const uint8_t*> TreeReadVarint64FromArray(
    const uint8_t* buffer, uint64_t* value) {
  uint64_t first8;
  std::memcpy(&first8, buffer, sizeof(first8));

  uint64_t term_mask = (~first8) & 0x8080808080808080ULL;
  if (__builtin_expect(term_mask != 0, 1)) {
    int ctz = __builtin_ctzll(term_mask);
    int len = (ctz >> 3) + 1;
    uint64_t step1 = (first8 & 0x007f007f007f007fULL) | ((first8 & 0x7f007f007f007f00ULL) >> 1);
    uint64_t step2 = (step1 & 0x00003fff00003fffULL) | ((step1 & 0x3fff00003fff0000ULL) >> 2);
    uint64_t step3 = (step2 & 0x0fffffffULL) | ((step2 >> 4) & (0x0fffffffULL << 28));
    uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (7 * len)) - 1);
    *value = step3 & mask;
    return {true, buffer + len};
  }

  if (buffer[8] < 128) {
    return {true, BaselineDecodeVarint64KnownSize<9>(buffer, value)};
  } else if (buffer[9] < 128) {
    return {true, BaselineDecodeVarint64KnownSize<10>(buffer, value)};
  }
  return {false, buffer + 11};
}

inline std::pair<bool, const uint8_t*> TreeReadVarint32FromArray(
    uint32_t first_byte, const uint8_t* buffer, uint32_t* value) {
  uint64_t first8;
  std::memcpy(&first8, buffer, sizeof(first8));

  uint64_t term_mask = (~first8) & 0x8080808080808080ULL;
  if (__builtin_expect(term_mask != 0, 1)) {
    int ctz = __builtin_ctzll(term_mask);
    int len = (ctz >> 3) + 1;
    uint64_t step1 = (first8 & 0x007f007f007f007fULL) | ((first8 & 0x7f007f007f007f00ULL) >> 1);
    uint64_t step2 = (step1 & 0x00003fff00003fffULL) | ((step1 & 0x3fff00003fff0000ULL) >> 2);
    uint64_t step3 = (step2 & 0x0fffffffULL) | ((step2 >> 4) & (0x0fffffffULL << 28));
    uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (7 * len)) - 1);
    *value = static_cast<uint32_t>(step3 & mask);
    return {true, buffer + len};
  }

  uint64_t val64;
  auto res = BaselineReadVarint64FromArray(buffer, &val64);
  if (res.first) {
    *value = static_cast<uint32_t>(val64);
  }
  return res;
}

int EncodeVarint(uint64_t value, uint8_t* out) {
  int count = 0;
  while (value >= 0x80) {
    out[count++] = static_cast<uint8_t>(value | 0x80);
    value >>= 7;
  }
  out[count++] = static_cast<uint8_t>(value);
  return count;
}

int main() {
  printf("================ Protobuf Varint Decoding Benchmark ================\n");

  std::mt19937_64 rng(42);
  const size_t NUM_ITEMS = 2000000;
  std::vector<uint64_t> raw_values(NUM_ITEMS);
  std::vector<uint8_t> encoded_data;
  encoded_data.reserve(NUM_ITEMS * 10 + 64);
  std::vector<size_t> offsets(NUM_ITEMS);

  for (size_t i = 0; i < NUM_ITEMS; i++) {
    uint32_t roll = rng() % 100;
    uint64_t val;
    if (roll < 40) {
      val = 128 + (rng() % 16256); // 2 bytes
    } else if (roll < 70) {
      val = 16384 + (rng() % 2000000); // 3 bytes
    } else if (roll < 90) {
      val = 2097152 + (rng() % 200000000); // 4 bytes
    } else if (roll < 98) {
      val = 268435456ULL + (rng() % 4000000000ULL); // 5 bytes
    } else {
      val = (1ULL << 35) + (rng() % (1ULL << 60)); // 6-10 bytes
    }
    raw_values[i] = val;
    offsets[i] = encoded_data.size();
    uint8_t buf[16];
    int len = EncodeVarint(val, buf);
    encoded_data.insert(encoded_data.end(), buf, buf + len);
  }
  for (int p = 0; p < 32; p++) encoded_data.push_back(0);

  printf("Generated %zu varints (%zu bytes, avg %.2f bytes/varint)\n",
         NUM_ITEMS, encoded_data.size(), (double)encoded_data.size() / NUM_ITEMS);

  // 1. Correctness Verification
  printf("\n[1] Verifying Correctness for PEXT and Portable Tree...\n");
  for (size_t i = 0; i < NUM_ITEMS; i++) {
    const uint8_t* ptr = encoded_data.data() + offsets[i];
    uint64_t v_base = 0, v_pext = 0, v_tree = 0;
    auto r_base = BaselineReadVarint64FromArray(ptr, &v_base);
    auto r_pext = PextReadVarint64FromArray(ptr, &v_pext);
    auto r_tree = TreeReadVarint64FromArray(ptr, &v_tree);

    if (!r_base.first || !r_pext.first || !r_tree.first ||
        v_base != raw_values[i] || v_pext != raw_values[i] || v_tree != raw_values[i] ||
        r_pext.second != r_base.second || r_tree.second != r_base.second) {
      printf("Varint64 MISMATCH at %zu! expected=%lu base=%lu pext=%lu tree=%lu\n",
             i, raw_values[i], v_base, v_pext, v_tree);
      return 1;
    }

    if (raw_values[i] <= 0xffffffffULL) {
      uint32_t v32_base = 0, v32_pext = 0, v32_tree = 0;
      auto r32_base = BaselineReadVarint32FromArray(*ptr, ptr, &v32_base);
      auto r32_pext = PextReadVarint32FromArray(*ptr, ptr, &v32_pext);
      auto r32_tree = TreeReadVarint32FromArray(*ptr, ptr, &v32_tree);
      if (!r32_base.first || !r32_pext.first || !r32_tree.first ||
          v32_base != (uint32_t)raw_values[i] || v32_pext != (uint32_t)raw_values[i] || v32_tree != (uint32_t)raw_values[i]) {
        printf("Varint32 MISMATCH at %zu! expected=%u base=%u pext=%u tree=%u\n",
               i, (uint32_t)raw_values[i], v32_base, v32_pext, v32_tree);
        return 1;
      }
    }
  }
  printf(">>> 100%% VERIFIED: ALL %zu Varints match exactly byte-for-byte across all decoders!\n", NUM_ITEMS);

  // 2. Performance Benchmark: Varint64
  const int ITERS = 5;
  printf("\n[2] Benchmarking ReadVarint64FromArray (%zu items x %d iterations = %zu total ops)...\n",
         NUM_ITEMS, ITERS, NUM_ITEMS * ITERS);

  // Baseline
  uint64_t dummy = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint64_t val;
      BaselineReadVarint64FromArray(encoded_data.data() + offsets[i], &val);
      dummy += val;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_base64 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Portable Tree
  dummy = 0;
  t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint64_t val;
      TreeReadVarint64FromArray(encoded_data.data() + offsets[i], &val);
      dummy += val;
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_tree64 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // BMI2 PEXT
  dummy = 0;
  t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint64_t val;
      PextReadVarint64FromArray(encoded_data.data() + offsets[i], &val);
      dummy += val;
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_pext64 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("Baseline Varint64:      %7.2f ms | %6.2f M ops/s | %5.2f ns/op\n",
         ms_base64, (NUM_ITEMS * ITERS) / (ms_base64 * 1000.0), ms_base64 * 1e6 / (NUM_ITEMS * ITERS));
  printf("Portable Tree Varint64: %7.2f ms | %6.2f M ops/s | %5.2f ns/op -> %.2fx faster!\n",
         ms_tree64, (NUM_ITEMS * ITERS) / (ms_tree64 * 1000.0), ms_tree64 * 1e6 / (NUM_ITEMS * ITERS), ms_base64 / ms_tree64);
  printf("BMI2 PEXT Varint64:     %7.2f ms | %6.2f M ops/s | %5.2f ns/op -> %.2fx faster!\n",
         ms_pext64, (NUM_ITEMS * ITERS) / (ms_pext64 * 1000.0), ms_pext64 * 1e6 / (NUM_ITEMS * ITERS), ms_base64 / ms_pext64);

  // 3. Performance Benchmark: Varint32
  printf("\n[3] Benchmarking ReadVarint32FromArray (%zu items x %d iterations = %zu total ops)...\n",
         NUM_ITEMS, ITERS, NUM_ITEMS * ITERS);

  dummy = 0;
  t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint32_t val;
      const uint8_t* ptr = encoded_data.data() + offsets[i];
      BaselineReadVarint32FromArray(*ptr, ptr, &val);
      dummy += val;
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_base32 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  dummy = 0;
  t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint32_t val;
      const uint8_t* ptr = encoded_data.data() + offsets[i];
      TreeReadVarint32FromArray(*ptr, ptr, &val);
      dummy += val;
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_tree32 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  dummy = 0;
  t0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < ITERS; iter++) {
    for (size_t i = 0; i < NUM_ITEMS; i++) {
      uint32_t val;
      const uint8_t* ptr = encoded_data.data() + offsets[i];
      PextReadVarint32FromArray(*ptr, ptr, &val);
      dummy += val;
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  DoNotOptimize(dummy);
  double ms_pext32 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("Baseline Varint32:      %7.2f ms | %6.2f M ops/s | %5.2f ns/op\n",
         ms_base32, (NUM_ITEMS * ITERS) / (ms_base32 * 1000.0), ms_base32 * 1e6 / (NUM_ITEMS * ITERS));
  printf("Portable Tree Varint32: %7.2f ms | %6.2f M ops/s | %5.2f ns/op -> %.2fx faster!\n",
         ms_tree32, (NUM_ITEMS * ITERS) / (ms_tree32 * 1000.0), ms_tree32 * 1e6 / (NUM_ITEMS * ITERS), ms_base32 / ms_tree32);
  printf("BMI2 PEXT Varint32:     %7.2f ms | %6.2f M ops/s | %5.2f ns/op -> %.2fx faster!\n",
         ms_pext32, (NUM_ITEMS * ITERS) / (ms_pext32 * 1000.0), ms_pext32 * 1e6 / (NUM_ITEMS * ITERS), ms_base32 / ms_pext32);

  return 0;
}
