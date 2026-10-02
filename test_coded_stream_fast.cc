#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/parse_context.h"

using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::internal::VarintParse;

int EncodeVarint(uint64_t value, uint8_t* out) {
  int count = 0;
  while (value >= 0x80) {
    out[count++] = static_cast<uint8_t>(value | 0x80);
    value >>= 7;
  }
  out[count++] = static_cast<uint8_t>(value);
  return count;
}

void TestVarint32() {
  printf("[TEST] Testing CodedInputStream::ReadVarint32...\n");
  std::mt19937_64 rng(12345);

  std::vector<uint32_t> test_values = {
      0, 1, 2, 127, 128, 129, 255, 256, 16383, 16384, 16385,
      2097151, 2097152, 268435455, 268435456, 0x7fffffff, 0x80000000, 0xffffffff
  };

  for (int i = 0; i < 50000; i++) {
    test_values.push_back(static_cast<uint32_t>(rng()));
  }

  std::vector<uint8_t> buffer;
  buffer.reserve(test_values.size() * 5 + 64);
  for (uint32_t v : test_values) {
    uint8_t buf[16];
    int len = EncodeVarint(v, buf);
    buffer.insert(buffer.end(), buf, buf + len);
  }

  ArrayInputStream raw_input(buffer.data(), buffer.size());
  CodedInputStream input(&raw_input);

  for (size_t i = 0; i < test_values.size(); i++) {
    uint32_t decoded = 0;
    bool ok = input.ReadVarint32(&decoded);
    if (!ok || decoded != test_values[i]) {
      printf("FAIL ReadVarint32 at index %zu: expected %u, got %u (ok=%d)\n",
             i, test_values[i], decoded, ok);
      assert(false);
    }
  }
  printf("  PASS: 50,018 Varint32 tests verified!\n");
}

void TestVarint64() {
  printf("[TEST] Testing CodedInputStream::ReadVarint64...\n");
  std::mt19937_64 rng(67890);

  std::vector<uint64_t> test_values = {
      0, 1, 2, 127, 128, 255, 16383, 16384, 2097151, 2097152,
      268435455, 268435456, 0xffffffffULL, 0x100000000ULL,
      0x7fffffffffffffffULL, 0x8000000000000000ULL, 0xffffffffffffffffULL
  };

  for (int i = 0; i < 50000; i++) {
    test_values.push_back(rng());
  }

  std::vector<uint8_t> buffer;
  buffer.reserve(test_values.size() * 10 + 64);
  for (uint64_t v : test_values) {
    uint8_t buf[16];
    int len = EncodeVarint(v, buf);
    buffer.insert(buffer.end(), buf, buf + len);
  }

  ArrayInputStream raw_input(buffer.data(), buffer.size());
  CodedInputStream input(&raw_input);

  for (size_t i = 0; i < test_values.size(); i++) {
    uint64_t decoded = 0;
    bool ok = input.ReadVarint64(&decoded);
    if (!ok || decoded != test_values[i]) {
      printf("FAIL ReadVarint64 at index %zu: expected %lu, got %lu (ok=%d)\n",
             i, test_values[i], decoded, ok);
      assert(false);
    }
  }
  printf("  PASS: 50,017 Varint64 tests verified!\n");
}

void TestReadTag() {
  printf("[TEST] Testing CodedInputStream::ReadTag...\n");
  std::vector<uint32_t> tags = {
      (1 << 3) | 0, (2 << 3) | 2, (15 << 3) | 1, (1000 << 3) | 2,
      (50000 << 3) | 0, (5000000 << 3) | 5
  };

  std::vector<uint8_t> buffer;
  for (uint32_t t : tags) {
    uint8_t buf[16];
    int len = EncodeVarint(t, buf);
    buffer.insert(buffer.end(), buf, buf + len);
  }

  ArrayInputStream raw_input(buffer.data(), buffer.size());
  CodedInputStream input(&raw_input);

  for (size_t i = 0; i < tags.size(); i++) {
    uint32_t tag = input.ReadTag();
    assert(tag == tags[i]);
  }
  printf("  PASS: All tags match!\n");
}

void TestBufferBoundaries() {
  printf("[TEST] Testing buffer boundary conditions (sizes 1 to 15)...\n");
  for (int size = 1; size <= 15; size++) {
    uint64_t test_val = 0;
    if (size == 1) test_val = 42;
    else if (size == 2) test_val = 300;
    else if (size == 3) test_val = 70000;
    else if (size == 4) test_val = 10000000;
    else if (size == 5) test_val = 2000000000ULL;
    else test_val = 1ULL << (7 * (size - 1));

    uint8_t buf[16] = {};
    int len = EncodeVarint(test_val, buf);
    if (len > size) continue;

    ArrayInputStream raw_input(buf, size);
    CodedInputStream input(&raw_input);

    uint64_t decoded64 = 0;
    bool ok64 = input.ReadVarint64(&decoded64);
    assert(ok64 && decoded64 == test_val);

    if (test_val <= 0xffffffffULL) {
      ArrayInputStream raw_input32(buf, size);
      CodedInputStream input32(&raw_input32);
      uint32_t decoded32 = 0;
      bool ok32 = input32.ReadVarint32(&decoded32);
      assert(ok32 && decoded32 == static_cast<uint32_t>(test_val));
    }
  }
  printf("  PASS: Buffer boundary tests passed!\n");
}

void TestCorruptVarints() {
  printf("[TEST] Testing corrupt varints (> 10 continuation bytes)...\n");
  uint8_t corrupt_buf[16];
  for (int i = 0; i < 16; i++) corrupt_buf[i] = 0x80 | (i & 0x7f);

  ArrayInputStream raw_input(corrupt_buf, sizeof(corrupt_buf));
  CodedInputStream input(&raw_input);

  uint64_t val64 = 0;
  bool ok64 = input.ReadVarint64(&val64);
  assert(!ok64);

  ArrayInputStream raw_input32(corrupt_buf, sizeof(corrupt_buf));
  CodedInputStream input32(&raw_input32);
  uint32_t val32 = 0;
  bool ok32 = input32.ReadVarint32(&val32);
  assert(!ok32);

  printf("  PASS: Corrupt varints rejected properly!\n");
}

void TestVarintParseContext() {
  printf("[TEST] Testing ParseContext::VarintParse (internal::VarintParse)...\n");
  std::mt19937_64 rng(99999);

  std::vector<uint64_t> test_values = {
      0, 1, 127, 128, 255, 16383, 16384, 2097151, 2097152,
      268435455, 268435456, 0xffffffffULL, 0x100000000ULL,
      0x7fffffffffffffffULL, 0xffffffffffffffffULL
  };
  for (int i = 0; i < 20000; i++) {
    test_values.push_back(rng());
  }

  for (uint64_t val : test_values) {
    uint8_t buf[32] = {};
    int len = EncodeVarint(val, buf);

    // 64-bit VarintParse
    uint64_t parsed64 = 0;
    const char* ptr64 = VarintParse(reinterpret_cast<const char*>(buf), &parsed64);
    assert(ptr64 == reinterpret_cast<const char*>(buf) + len);
    assert(parsed64 == val);

    // 32-bit VarintParse (if <= 32-bit max)
    if (val <= 0xffffffffULL) {
      uint32_t parsed32 = 0;
      const char* ptr32 = VarintParse(reinterpret_cast<const char*>(buf), &parsed32);
      assert(ptr32 == reinterpret_cast<const char*>(buf) + len);
      assert(parsed32 == static_cast<uint32_t>(val));
    }
  }
  printf("  PASS: 20,015 VarintParse tests verified successfully!\n");
}

int main() {
  printf("================ PROTUBUF FAST VARINT TEST SUITE ================\n");
  TestVarint32();
  TestVarint64();
  TestReadTag();
  TestBufferBoundaries();
  TestCorruptVarints();
  TestVarintParseContext();
  printf(">>> 100%% ALL FAST VARINT TESTS PASSED SUCCESSFULLY! <<<\n");
  return 0;
}
