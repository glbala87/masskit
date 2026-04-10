#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace lcms {
namespace io {

/**
 * @brief Base64 encoding/decoding utilities.
 *
 * Provides functions for Base64 encoding and decoding, commonly used
 * in mzML and mzXML files for binary data storage.
 */
class Base64 {
public:
    /**
     * @brief Decode a Base64 encoded string to binary data.
     *
     * @param input Base64 encoded string
     * @return Decoded binary data
     * @throws std::invalid_argument if input contains invalid characters
     */
    static std::vector<std::uint8_t> decode(const std::string& input);

    /**
     * @brief Encode binary data to Base64 string.
     *
     * @param data Binary data
     * @return Base64 encoded string
     */
    static std::string encode(const std::vector<std::uint8_t>& data);

    /**
     * @brief Encode binary data to Base64 string.
     *
     * @param data Pointer to binary data
     * @param length Length of data in bytes
     * @return Base64 encoded string
     */
    static std::string encode(const std::uint8_t* data, std::size_t length);

    /**
     * @brief Decode Base64 to array of 64-bit floats.
     *
     * @param input Base64 encoded string
     * @param little_endian true if data is little-endian
     * @return Vector of decoded doubles
     */
    static std::vector<double> decodeFloat64(const std::string& input,
                                             bool little_endian = true);

    /**
     * @brief Decode Base64 to array of 32-bit floats.
     *
     * @param input Base64 encoded string
     * @param little_endian true if data is little-endian
     * @return Vector of decoded floats
     */
    static std::vector<float> decodeFloat32(const std::string& input,
                                            bool little_endian = true);

    /**
     * @brief Encode array of 64-bit floats to Base64.
     *
     * @param data Vector of doubles
     * @param little_endian true to encode as little-endian
     * @return Base64 encoded string
     */
    static std::string encodeFloat64(const std::vector<double>& data,
                                     bool little_endian = true);

    /**
     * @brief Encode array of 32-bit floats to Base64.
     *
     * @param data Vector of floats
     * @param little_endian true to encode as little-endian
     * @return Base64 encoded string
     */
    static std::string encodeFloat32(const std::vector<float>& data,
                                     bool little_endian = true);

    /**
     * @brief Check if a string is valid Base64.
     *
     * @param input String to check
     * @return true if valid Base64
     */
    static bool isValid(const std::string& input);

    /**
     * @brief Compute the decoded size of Base64 data.
     *
     * @param input Base64 string
     * @return Decoded size in bytes
     */
    static std::size_t decodedSize(const std::string& input);

private:
    static const char encoding_table_[];
    static const int decoding_table_[];

    static void swapEndian(void* data, std::size_t element_size,
                           std::size_t count);
};

#ifdef MASSKIT_HAS_ZLIB
/**
 * @brief Zlib compression/decompression utilities.
 */
class Zlib {
public:
    /**
     * @brief Decompress zlib compressed data.
     *
     * @param input Compressed data
     * @return Decompressed data
     * @throws std::runtime_error if decompression fails
     */
    static std::vector<std::uint8_t> decompress(
        const std::vector<std::uint8_t>& input);

    /**
     * @brief Compress data using zlib.
     *
     * @param input Uncompressed data
     * @param level Compression level (0-9, default 6)
     * @return Compressed data
     */
    static std::vector<std::uint8_t> compress(
        const std::vector<std::uint8_t>& input, int level = 6);

    /**
     * @brief Decompress and decode Base64 to doubles.
     *
     * @param base64_input Base64 encoded, zlib compressed data
     * @param little_endian true if data is little-endian
     * @return Vector of decoded doubles
     */
    static std::vector<double> decompressFloat64(const std::string& base64_input,
                                                 bool little_endian = true);

    /**
     * @brief Decompress and decode Base64 to floats.
     *
     * @param base64_input Base64 encoded, zlib compressed data
     * @param little_endian true if data is little-endian
     * @return Vector of decoded floats
     */
    static std::vector<float> decompressFloat32(const std::string& base64_input,
                                                bool little_endian = true);
};
#endif // MASSKIT_HAS_ZLIB

} // namespace io
} // namespace lcms
