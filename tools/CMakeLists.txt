# "tools" are supposed to be packaged with LDC.
# (in contrast to "utils" which are only used for building / testing)

#############################################################################
# Build ldc-prune-cache
set(LDCPRUNECACHE_EXE ldc-prune-cache)
set(LDCPRUNECACHE_EXE ${LDCPRUNECACHE_EXE} PARENT_SCOPE) # needed for correctly populating lit.site.cfg.in
set(LDCPRUNECACHE_EXE_NAME ${PROGRAM_PREFIX}${LDCPRUNECACHE_EXE}${PROGRAM_SUFFIX})
set(LDCPRUNECACHE_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDCPRUNECACHE_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(LDCPRUNECACHE_D_SRC
    ${PROJECT_SOURCE_DIR}/tools/ldc-prune-cache.d
    ${PROJECT_SOURCE_DIR}/driver/cache_pruning.d
)
build_d_executable(
    "${LDCPRUNECACHE_EXE}"
    "${LDCPRUNECACHE_EXE_FULL}"
    "${LDCPRUNECACHE_D_SRC}"
    "${DFLAGS_BUILD_TYPE}"
    ""
    ""
    ""
    ${COMPILE_D_MODULES_SEPARATELY}
)
install(PROGRAMS ${LDCPRUNECACHE_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#############################################################################
# Build ldc-profdata for converting profile data formats (source version depends on LLVM version)
set(LDCPROFDATA_SRC ldc-profdata/llvm-profdata-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LDCPROFDATA_SRC})
    add_executable(ldc-profdata ${LDCPROFDATA_SRC})
    set_target_properties(
        ldc-profdata PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS}"
    )
    target_link_libraries(ldc-profdata ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
    install(TARGETS ldc-profdata DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
