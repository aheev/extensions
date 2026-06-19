if (${BUILD_EXTENSION_TESTS})
    # Lance extension tests require the lance extension to be built
    add_lbug_test(lance_api_test
        lance_node_table_test.cpp
        lance_rel_table_test.cpp
        lance_error_scenarios_test.cpp
        lance_vector_search_test.cpp
    )
    target_link_libraries(lance_api_test PRIVATE lbug_${EXTENSION_LIB_NAME}_extension)
    target_include_directories(lance_api_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/include
        ${CMAKE_BINARY_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/extension/lance/src/include
        ${LANCE_C_ROOT}/include
    )
endif()
