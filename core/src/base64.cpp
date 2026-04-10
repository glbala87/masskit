#include "lcms/io/base64.hpp"
#include <cstring>
#include <algorithm>

#ifdef MASSKIT_HAS_ZLIB
#include <zlib.h>
#endif

namespace lcms {
namespace io {

// Base64 encoding table
const char Base64::encoding_table_[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Base64 decoding table (-1 = invalid, -2 = whitespace/padding)
const int Base64::decoding_table_[] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,-2,-1,-1,  // 0-15
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 16-31
    -2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,  // 32-47 (space, +, /)
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,  // 48-63 (0-9, =)
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,  // 64-79 (A-O)
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,  // 80-95 (P-Z)
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,  // 96-111 (a-o)
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,  // 112-127 (p-z)
};

std::vector<std::uint8_t> Base64::decode(const std::string& input) {
    if (input.empty()) {
        return {};
    }

    // Estimate output size (slightly over-estimated)
    std::size_t output_length = (input.size() * 3) / 4;
    std::vector<std::uint8_t> output;
    output.reserve(output_length);

    std::uint32_t buffer = 0;
    int bits_in_buffer = 0;

    for (char c : input) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 128) {
            throw std::invalid_argument("Invalid Base64 character");
        }

        int value = decoding_table_[uc];
        if (value == -1) {
            throw std::invalid_argument("Invalid Base64 character");
        }
        if (value == -2) {
            continue;  // Skip whitespace and padding
        }

        buffer = (buffer << 6) | static_cast<std::uint32_t>(value);
        bits_in_buffer += 6;

        if (bits_in_buffer >= 8) {
            bits_in_buffer -= 8;
            output.push_back(static_cast<std::uint8_t>((buffer >> bits_in_buffer) & 0xFF));
        }
    }

    return output;
}

std::string Base64::encode(const std::vector<std::uint8_t>& data) {
    return encode(data.data(), data.size());
}

std::string Base64::encode(const std::uint8_t* data, std::size_t length) {
    if (length == 0) {
        return {};
    }

    std::size_t output_length = 4 * ((length + 2) / 3);
    std::string output;
    output.reserve(output_length);

    std::size_t i = 0;
    while (i + 3 <= length) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i+1]) << 8) |
                          static_cast<std::uint32_t>(data[i+2]);

        output += encoding_table_[(n >> 18) & 0x3F];
        output += encoding_table_[(n >> 12) & 0x3F];
        output += encoding_table_[(n >> 6) & 0x3F];
        output += encoding_table_[n & 0x3F];

        i += 3;
    }

    // Handle remaining bytes
    if (length - i == 1) {
        std::uint32_t n = static_cast<std::uint32_t>(data[i]) << 16;
        output += encoding_table_[(n >> 18) & 0x3F];
        output += encoding_table_[(n >> 12) & 0x3F];
        output += '=';
        output += '=';
    } else if (length - i == 2) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i+1]) << 8);
        output += encoding_table_[(n >> 18) & 0x3F];
        output += encoding_table_[(n >> 12) & 0x3F];
        output += encoding_table_[(n >> 6) & 0x3F];
        output += '=';
    }

    return output;
}

void Base64::swapEndian(void* data, std::size_t element_size, std::size_t count) {
    auto* bytes = static_cast<std::uint8_t*>(data);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint8_t* element = bytes + i * element_size;
        for (std::size_t j = 0; j < element_size / 2; ++j) {
            std::swap(element[j], element[element_size - 1 - j]);
        }
    }
}

std::vector<double> Base64::decodeFloat64(const std::string& input,
                                          bool little_endian) {
    auto binary = decode(input);

    if (binary.size() % 8 != 0) {
        throw std::invalid_argument("Invalid data size for float64");
    }

    std::size_t count = binary.size() / 8;
    std::vector<double> result(count);

    std::memcpy(result.data(), binary.data(), binary.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        swapEndian(result.data(), sizeof(double), count);
    }

    return result;
}

std::vector<float> Base64::decodeFloat32(const std::string& input,
                                         bool little_endian) {
    auto binary = decode(input);

    if (binary.size() % 4 != 0) {
        throw std::invalid_argument("Invalid data size for float32");
    }

    std::size_t count = binary.size() / 4;
    std::vector<float> result(count);

    std::memcpy(result.data(), binary.data(), binary.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        swapEndian(result.data(), sizeof(float), count);
    }

    return result;
}

std::string Base64::encodeFloat64(const std::vector<double>& data,
                                  bool little_endian) {
    if (data.empty()) {
        return {};
    }

    std::vector<std::uint8_t> binary(data.size() * sizeof(double));
    std::memcpy(binary.data(), data.data(), binary.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        swapEndian(binary.data(), sizeof(double), data.size());
    }

    return encode(binary);
}

std::string Base64::encodeFloat32(const std::vector<float>& data,
                                  bool little_endian) {
    if (data.empty()) {
        return {};
    }

    std::vector<std::uint8_t> binary(data.size() * sizeof(float));
    std::memcpy(binary.data(), data.data(), binary.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        swapEndian(binary.data(), sizeof(float), data.size());
    }

    return encode(binary);
}

bool Base64::isValid(const std::string& input) {
    for (char c : input) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 128) return false;
        if (decoding_table_[uc] == -1) return false;
    }
    return true;
}

std::size_t Base64::decodedSize(const std::string& input) {
    if (input.empty()) return 0;

    std::size_t len = input.size();

    // Count padding
    std::size_t padding = 0;
    if (len >= 1 && input[len-1] == '=') padding++;
    if (len >= 2 && input[len-2] == '=') padding++;

    return (len * 3) / 4 - padding;
}

#ifdef MASSKIT_HAS_ZLIB
std::vector<std::uint8_t> Zlib::decompress(const std::vector<std::uint8_t>& input) {
    if (input.empty()) {
        return {};
    }

    // Estimate output size (start with 4x input size)
    std::vector<std::uint8_t> output;
    std::size_t output_size = input.size() * 4;
    output.resize(output_size);

    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(input.data());
    stream.avail_in = static_cast<uInt>(input.size());
    stream.next_out = output.data();
    stream.avail_out = static_cast<uInt>(output.size());

    if (inflateInit(&stream) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib decompression");
    }

    int result;
    while ((result = inflate(&stream, Z_NO_FLUSH)) != Z_STREAM_END) {
        if (result == Z_OK) {
            // Need more output space
            std::size_t current = output.size() - stream.avail_out;
            output_size *= 2;
            output.resize(output_size);
            stream.next_out = output.data() + current;
            stream.avail_out = static_cast<uInt>(output_size - current);
        } else {
            inflateEnd(&stream);
            throw std::runtime_error("Zlib decompression failed");
        }
    }

    output.resize(output.size() - stream.avail_out);
    inflateEnd(&stream);

    return output;
}

std::vector<std::uint8_t> Zlib::compress(const std::vector<std::uint8_t>& input,
                                          int level) {
    if (input.empty()) {
        return {};
    }

    // Worst case compressed size
    std::size_t output_size = compressBound(static_cast<uLong>(input.size()));
    std::vector<std::uint8_t> output(output_size);

    uLongf compressed_size = static_cast<uLongf>(output_size);
    if (compress2(output.data(), &compressed_size,
                  input.data(), static_cast<uLong>(input.size()), level) != Z_OK) {
        throw std::runtime_error("Zlib compression failed");
    }

    output.resize(compressed_size);
    return output;
}

std::vector<double> Zlib::decompressFloat64(const std::string& base64_input,
                                             bool little_endian) {
    auto binary = Base64::decode(base64_input);
    auto decompressed = decompress(binary);

    if (decompressed.size() % 8 != 0) {
        throw std::invalid_argument("Invalid decompressed data size for float64");
    }

    std::size_t count = decompressed.size() / 8;
    std::vector<double> result(count);
    std::memcpy(result.data(), decompressed.data(), decompressed.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        Base64::swapEndian(result.data(), sizeof(double), count);
    }

    return result;
}

std::vector<float> Zlib::decompressFloat32(const std::string& base64_input,
                                            bool little_endian) {
    auto binary = Base64::decode(base64_input);
    auto decompressed = decompress(binary);

    if (decompressed.size() % 4 != 0) {
        throw std::invalid_argument("Invalid decompressed data size for float32");
    }

    std::size_t count = decompressed.size() / 4;
    std::vector<float> result(count);
    std::memcpy(result.data(), decompressed.data(), decompressed.size());

    // Check system endianness and swap if needed
    union {
        std::uint32_t i;
        char c[4];
    } endian_test = {0x01020304};

    bool system_is_little = (endian_test.c[0] == 4);

    if (little_endian != system_is_little) {
        Base64::swapEndian(result.data(), sizeof(float), count);
    }

    return result;
}
#endif // MASSKIT_HAS_ZLIB

} // namespace io
} // namespace lcms
