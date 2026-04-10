cmake_minimum_required(VERSION 3.14)
project(masskit_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(MASSKIT_BUILD_TESTS "Build unit tests" ON)
option(MASSKIT_BUILD_PYTHON "Build Python bindings" OFF)

# Find dependencies
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PUGIXML QUIET pugixml)
endif()

# If pugixml not found via pkg-config, try find_package
if(NOT PUGIXML_FOUND)
    find_package(pugixml QUIET)
endif()

# ZLIB for compression support
find_package(ZLIB QUIET)

# Eigen for linear algebra
find_package(Eigen3 QUIET)

# Core library sources
set(LCMS_SOURCES
    src/spectrum.cpp
    src/chromatogram.cpp
    src/peak.cpp
    src/feature_map.cpp
    src/ms_experiment.cpp
    src/mzml_reader.cpp
    src/mzxml_reader.cpp
    src/base64.cpp
    src/isotope_detection.cpp
    src/spectral_matching.cpp
    src/quantification.cpp
    src/parallel.cpp
    src/indexed_reader.cpp
)

# Create the core library
add_library(lcms_core ${LCMS_SOURCES})
add_library(lcms::core ALIAS lcms_core)

target_include_directories(lcms_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
if(PUGIXML_FOUND)
    target_link_libraries(lcms_core PUBLIC ${PUGIXML_LIBRARIES})
    target_include_directories(lcms_core PUBLIC ${PUGIXML_INCLUDE_DIRS})
    target_compile_definitions(lcms_core PUBLIC MASSKIT_HAS_PUGIXML)
endif()

if(ZLIB_FOUND)
    target_link_libraries(lcms_core PUBLIC ZLIB::ZLIB)
    target_compile_definitions(lcms_core PUBLIC MASSKIT_HAS_ZLIB)
endif()

if(Eigen3_FOUND)
    target_link_libraries(lcms_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(lcms_core PUBLIC MASSKIT_HAS_EIGEN)
endif()

# Threading support for parallel processing
find_package(Threads REQUIRED)
target_link_libraries(lcms_core PUBLIC Threads::Threads)

# Compiler warnings
target_compile_options(lcms_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Python bindings
if(MASSKIT_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(_masskit_core bindings/python_bindings.cpp)
        target_link_libraries(_masskit_core PRIVATE lcms_core)
        target_compile_definitions(_masskit_core PRIVATE MASSKIT_BUILD_PYTHON)
        message(STATUS "Building Python bindings with pybind11")
    else()
        message(WARNING "pybind11 not found - Python bindings will not be built")
    endif()
endif()

# Tests
if(MASSKIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS lcms_core
    EXPORT lcms_coreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/lcms
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT lcms_coreTargets
    FILE lcms_coreTargets.cmake
    NAMESPACE lcms::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lcms_core
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lcms_coreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lcms_coreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lcms_coreConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lcms_core
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lcms_coreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lcms_coreConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lcms_core
)
