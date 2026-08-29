/*
 * Focused adapter for CPU-001 differential verification only.
 *
 * Compile this file with the pinned llama.cpp ggml/src/ggml-quants.c and its
 * include directories. The block definitions and dequantizers are MIT-licensed
 * upstream code; this adapter only reads hex, calls them, and writes FP32 bytes.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml-quants.h"

static int hex_value(char character) {
  if (character >= '0' && character <= '9') return character - '0';
  if (character >= 'a' && character <= 'f') return character - 'a' + 10;
  if (character >= 'A' && character <= 'F') return character - 'A' + 10;
  return -1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fputs("usage: llama-quant-oracle q4_k|q6_k|q8_0 BLOCK_HEX\n", stderr);
    return 2;
  }
  const size_t expected =
      strcmp(argv[1], "q4_k") == 0
          ? sizeof(block_q4_K)
          : strcmp(argv[1], "q6_k") == 0
                ? sizeof(block_q6_K)
                : strcmp(argv[1], "q8_0") == 0 ? sizeof(block_q8_0) : 0;
  if (expected == 0 || strlen(argv[2]) != expected * 2) {
    fputs("invalid kind or block size\n", stderr);
    return 1;
  }
  union {
    block_q4_K q4;
    block_q6_K q6;
    block_q8_0 q8;
    uint8_t bytes[sizeof(block_q6_K)];
  } block;
  for (size_t index = 0; index < expected; ++index) {
    const int high = hex_value(argv[2][index * 2]);
    const int low = hex_value(argv[2][index * 2 + 1]);
    if (high < 0 || low < 0) {
      fputs("invalid hex\n", stderr);
      return 1;
    }
    block.bytes[index] = (uint8_t)((high << 4) | low);
  }
  float output[QK_K];
  if (strcmp(argv[1], "q4_k") == 0) {
    dequantize_row_q4_K(&block.q4, output, QK_K);
  } else if (strcmp(argv[1], "q6_k") == 0) {
    dequantize_row_q6_K(&block.q6, output, QK_K);
  } else {
    dequantize_row_q8_0(&block.q8, output, QK8_0);
  }
  static const char hex[] = "0123456789abcdef";
  const size_t output_count = strcmp(argv[1], "q8_0") == 0 ? QK8_0 : QK_K;
  for (size_t index = 0; index < output_count; ++index) {
    uint32_t bits = 0;
    memcpy(&bits, output + index, sizeof(bits));
    for (unsigned int byte = 0; byte < 4; ++byte) {
      const unsigned int value = (bits >> (byte * 8U)) & 255U;
      putchar(hex[value >> 4U]);
      putchar(hex[value & 15U]);
    }
  }
  putchar('\n');
  return 0;
}
