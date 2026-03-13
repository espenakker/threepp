# cmake/FetchWgpuNative.cmake
# Downloads pre-built wgpu-native binaries for the current platform.

include(FetchContent)

set(WGPU_NATIVE_VERSION "v27.0.4.0" CACHE STRING "wgpu-native release version")

# Determine platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_wgpu_os "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_wgpu_os "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_wgpu_os "windows")
else()
    message(FATAL_ERROR
        "wgpu-native: unsupported platform '${CMAKE_SYSTEM_NAME}'. "
        "Supported: Linux, Darwin (macOS), Windows.")
endif()

# Determine architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_wgpu_arch "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_wgpu_arch "aarch64")
else()
    message(FATAL_ERROR
        "wgpu-native: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
        "Supported: x86_64/AMD64, aarch64/arm64.")
endif()

# Construct archive filename
if(_wgpu_os STREQUAL "windows")
    set(_wgpu_archive "wgpu-${_wgpu_os}-${_wgpu_arch}-msvc-release.zip")
else()
    set(_wgpu_archive "wgpu-${_wgpu_os}-${_wgpu_arch}-release.zip")
endif()

set(_wgpu_url
    "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}/${_wgpu_archive}")

message(STATUS "wgpu-native: fetching ${_wgpu_archive} (${WGPU_NATIVE_VERSION})")

FetchContent_Declare(
    wgpu_native
    URL "${_wgpu_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(wgpu_native)

# Locate the static library inside the fetched content
# Prefer the static library to avoid runtime dependency on libwgpu_native.so
if(WIN32)
    set(_wgpu_lib_name "wgpu_native.lib")
else()
    set(_wgpu_lib_name "libwgpu_native.a")
endif()

find_file(_wgpu_native_lib
    NAMES "${_wgpu_lib_name}"
    PATHS "${wgpu_native_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(NOT _wgpu_native_lib)
    message(FATAL_ERROR
        "wgpu-native: library not found in fetched content at "
        "${wgpu_native_SOURCE_DIR}/lib. "
        "Archive may have an unexpected layout. URL was: ${_wgpu_url}")
endif()

# Export variables for the rest of the build
set(WGPU_INCLUDE_DIR "${wgpu_native_SOURCE_DIR}/include" CACHE INTERNAL
    "wgpu-native include directory")
set(WGPU_LIBRARY "${_wgpu_native_lib}" CACHE INTERNAL
    "wgpu-native library path")

message(STATUS "wgpu-native: include = ${WGPU_INCLUDE_DIR}")
message(STATUS "wgpu-native: library = ${WGPU_LIBRARY}")
