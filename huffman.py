import heapq
import os
import pickle
import time
import tracemalloc
from collections import Counter


class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    # Less-than comparison for min-heap ordering
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCompressor:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}

    def _build_frequency_table(self, text: str) -> dict:
        return dict(Counter(text))

    def _build_huffman_tree(self, freq_table: dict) -> HuffmanNode:
        heap = [HuffmanNode(char, freq) for char, freq in freq_table.items()]
        heapq.heapify(heap)

        # Edge case: file contains only one unique character
        if len(heap) == 1:
            only_node = heapq.heappop(heap)
            return HuffmanNode(freq=only_node.freq, left=only_node)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)

        return heap[0] if heap else None

    def _generate_codes_helper(self, node: HuffmanNode, current_code: str):
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_mapping[current_code] = node.char
            return

        self._generate_codes_helper(node.left, current_code + "0")
        self._generate_codes_helper(node.right, current_code + "1")

    def _pad_encoded_text(self, encoded_text: str) -> str:
        extra_padding = 8 - (len(encoded_text) % 8)
        encoded_text += "0" * extra_padding
        padded_info = f"{extra_padding:08b}"
        return padded_info + encoded_text

    def _get_byte_array(self, padded_encoded_text: str) -> bytearray:
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i : i + 8]
            b.append(int(byte, 2))
        return b

    def compress(self, input_path: str, output_path: str) -> dict:
        self.codes.clear()
        self.reverse_mapping.clear()

        tracemalloc.start()
        start_time = time.perf_counter()

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        if not text:
            raise ValueError("Input file is empty. Cannot compress an empty file.")

        freq_table = self._build_frequency_table(text)
        root = self._build_huffman_tree(freq_table)
        self._generate_codes_helper(root, "")

        encoded_text = "".join(self.codes[char] for char in text)
        padded_encoded_text = self._pad_encoded_text(encoded_text)
        byte_array = self._get_byte_array(padded_encoded_text)

        # Store frequency table in header so file is self-contained for decoding
        with open(output_path, "wb") as out_file:
            pickle.dump(freq_table, out_file)
            out_file.write(bytes(byte_array))

        elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = (1 - (compressed_size / original_size)) * 100

        return {
            "unique_chars": len(freq_table),
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio_pct": compression_ratio,
            "time_ms": elapsed_time,
            "peak_memory_kb": peak_memory / 1024,
        }

    def _remove_padding(self, padded_encoded_text: str) -> str:
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)
        encoded_text = padded_encoded_text[8:]
        return encoded_text[:-extra_padding] if extra_padding > 0 else encoded_text

    def decompress(self, compressed_path: str, output_path: str) -> dict:
        self.codes.clear()
        self.reverse_mapping.clear()

        tracemalloc.start()
        start_time = time.perf_counter()

        with open(compressed_path, "rb") as file:
            freq_table = pickle.load(file)
            compressed_data = file.read()

        root = self._build_huffman_tree(freq_table)
        self._generate_codes_helper(root, "")

        bit_string = "".join(f"{byte:08b}" for byte in compressed_data)
        encoded_text = self._remove_padding(bit_string)

        decoded_chars = []
        current_code = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_chars.append(self.reverse_mapping[current_code])
                current_code = ""

        decoded_text = "".join(decoded_chars)

        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(decoded_text)

        elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "time_ms": elapsed_time,
            "peak_memory_kb": peak_memory / 1024,
        }


def run_benchmark(sample_file="sample.txt"):
    # Create a sample file if one does not exist
    if not os.path.exists(sample_file):
        sample_content = (
            "Huffman coding is a lossless data compression algorithm. "
            "The idea is to assign variable-length codes to input characters, "
            "lengths of the assigned codes are based on the frequencies of corresponding characters. "
        ) * 50
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_content)

    compressed_file = "sample.huff"
    decompressed_file = "sample_decoded.txt"

    compressor = HuffmanCompressor()

    print("=" * 60)
    print("HUFFMAN CODING FILE COMPRESSION SYSTEM")
    print("=" * 60)

    comp_stats = compressor.compress(sample_file, compressed_file)
    print(f"\n[+] Compression Complete: '{sample_file}' -> '{compressed_file}'")
    print(f"    Unique Characters : {comp_stats['unique_chars']}")
    print(f"    Original Size     : {comp_stats['original_size_bytes']:,} bytes")
    print(f"    Compressed Size   : {comp_stats['compressed_size_bytes']:,} bytes")
    print(f"    Space Saved       : {comp_stats['compression_ratio_pct']:.2f}%")
    print(f"    Execution Time    : {comp_stats['time_ms']:.2f} ms")
    print(f"    Peak Memory Used  : {comp_stats['peak_memory_kb']:.2f} KB")

    print("\n[+] Generated Huffman Prefix Codes (Top 10 by shortest code):")
    sorted_codes = sorted(compressor.codes.items(), key=lambda x: len(x[1]))[:10]
    for char, code in sorted_codes:
        display_char = repr(char) if char in ("\n", "\t", " ") else char
        print(f"    {display_char:<6} -> {code}")

    decomp_stats = compressor.decompress(compressed_file, decompressed_file)
    print(f"\n[+] Decompression Complete: '{compressed_file}' -> '{decompressed_file}'")
    print(f"    Execution Time    : {decomp_stats['time_ms']:.2f} ms")
    print(f"    Peak Memory Used  : {decomp_stats['peak_memory_kb']:.2f} KB")

    # Verify lossless integrity
    with open(sample_file, "r", encoding="utf-8") as f1, open(
        decompressed_file, "r", encoding="utf-8"
    ) as f2:
        is_identical = f1.read() == f2.read()

    print(f"\n[+] Lossless Integrity Check: {'PASSED (100% Match)' if is_identical else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()


