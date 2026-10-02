#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <utility>

// Reference baseline decoder
template <size_t N>
inline const uint8_t* BaselineDecodeVarint64KnownSize(const uint8_t* buffer, uint64_t* value) {
  uint64_t result = static_cast<uint64_t>(buffer[N - 1]) << (7 * (N - 1));
  for (size_t i = 0, offset = 0; i < N - 1; i++, offset += 7) {
    result += static_cast<uint64_t>(buffer[i] - 0x80) << offset;
  }
  *value = result;
  return buffer + N;
}

inline std::pair<bool, const uint8_t*> BaselineReadVarint64(const uint8_t* buffer, size_t size, uint64_t* value) {
  if (size == 0) return {false, buffer};
  if (buffer[0] < 128) {
    *value = buffer[0];
    return {true, buffer + 1};
  }
  for (size_t n = 2; n <= 10 && n <= size; n++) {
    if (buffer[n - 1] < 128) {
      uint64_t res = static_cast<uint64_t>(buffer[n - 1]) << (7 * (n - 1));
      for (size_t i = 0, offset = 0; i < n - 1; i++, offset += 7) {
        res += static_cast<uint64_t>(buffer[i] - 0x80) << offset;
      }
      *value = res;
      return {true, buffer + n};
    }
  }
  return {false, buffer};
}

// Our branchless decoder with padded safety
inline std::pair<bool, const uint8_t*> FastReadVarint64(const uint8_t* buffer, size_t size, uint64_t* value) {
  if (size == 0) return {false, buffer};
  if (buffer[0] < 128) {
    *value = buffer[0];
    return {true, buffer + 1};
  }

  uint8_t padded[16] = {};
  size_t to_copy = (size < 16) ? size : 16;
  std::memcpy(padded, buffer, to_copy);

  uint64_t first8;
  std::memcpy(&first8, padded, sizeof(first8));

  uint64_t term_mask = (~first8) & 0x8080808080808080ULL;
  if (__builtin_expect(term_mask != 0, 1)) {
    int ctz = __builtin_ctzll(term_mask);
    size_t len = (ctz >> 3) + 1;
    if (len > size) return {false, buffer}; // Truncated input

    uint64_t step1 = (first8 & 0x007f007f007f007fULL) | ((first8 & 0x7f007f007f007f00ULL) >> 1);
    uint64_t step2 = (step1 & 0x00003fff00003fffULL) | ((step1 & 0x3fff00003fff0000ULL) >> 2);
    uint64_t step3 = (step2 & 0x0fffffffULL) | ((step2 >> 4) & (0x0fffffffULL << 28));
    uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (7 * len)) - 1);
    *value = step3 & mask;
    return {true, buffer + len};
  }

  // Length 9 or 10
  if (size >= 9 && padded[8] < 128) {
    *value = 0;
    for (size_t i = 0, offset = 0; i < 8; i++, offset += 7) {
      *value += static_cast<uint64_t>(padded[i] - 0x80) << offset;
    }
    *value += static_cast<uint64_t>(padded[8]) << 56;
    return {true, buffer + 9};
  } else if (size >= 10 && padded[9] < 128) {
    *value = 0;
    for (size_t i = 0, offset = 0; i < 9; i++, offset += 7) {
      *value += static_cast<uint64_t>(padded[i] - 0x80) << offset;
    }
    *value += static_cast<uint64_t>(padded[9]) << 63;
    return {true, buffer + 10};
  }

  return {false, buffer};
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;

  uint64_t val_base = 0;
  uint64_t val_fast = 0;

  auto res_base = BaselineReadVarint64(data, size, &val_base);
  auto res_fast = FastReadVarint64(data, size, &val_fast);

  // Both must agree on validity
  assert(res_base.first == res_fast.first);

  if (res_base.first) {
    // Both must agree on decoded value
    assert(val_base == val_fast);
    // Both must agree on bytes consumed
    assert(res_base.second == res_fast.second);
  }

  return 0;
}
