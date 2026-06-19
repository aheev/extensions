cmake_minimum_required(VERSION 3.22)

# ── lance-c dependency ────────────────────────────────────────────────────────
# Lance-c lives next to the ladybug repo at ../../lancedb/lance-c.
# We use add_subdirectory so its Corrosion / prebuilt CMake logic runs in-tree.
set(LANCE_C_ROOT "${PROJECT_SOURCE_DIR}/../../lancedb/lance-c"
    CACHE PATH "Path to the lance-c source tree")

if(NOT TARGET LanceC::lance_c)
    if(NOT EXISTS "${LANCE_C_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR
            "lance-c not found at ${LANCE_C_ROOT}. "
            "Set LANCE_C_ROOT to the lance-c repository path.")
    endif()
    # Use prebuilt if LANCE_C_PREBUILT_DIR is set, otherwise build from source
    if(LANCE_C_PREBUILT_DIR)
        set(LANCE_C_USE_PREBUILT ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(${LANCE_C_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/lance-c EXCLUDE_FROM_ALL)
endif()

# ── Extension source ──────────────────────────────────────────────────────────
include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${CMAKE_BINARY_DIR}/src/include
    src/include
    ${LANCE_C_ROOT}/include
)

set(LANCE_EXTENSION_OBJECT_FILES
    $<TARGET_OBJECTS:lance_extension_objects>
)

add_library(lance_extension_objects OBJECT
    src/lance_extension.cpp
    src/lance_node_table.cpp
    src/lance_rel_table.cpp
    src/lance_functions.cpp
)

target_include_directories(lance_extension_objects PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${CMAKE_BINARY_DIR}/src/include
    src/include
    ${LANCE_C_ROOT}/include
)

# ── Build the extension library ───────────────────────────────────────────────
build_extension_lib(${BUILD_STATIC_EXTENSION} "lance")

target_link_libraries(lbug_${EXTENSION_LIB_NAME}_extension
    PRIVATE
    LanceC::lance_c_static
)

# Platform-specific link dependencies for the Rust-backed lance-c library
if(UNIX AND NOT APPLE)
    target_link_libraries(lbug_${EXTENSION_LIB_NAME}_extension
        PRIVATE pthread dl m)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_EXTENSION_TESTS)
    add_subdirectory(test)
endif()
