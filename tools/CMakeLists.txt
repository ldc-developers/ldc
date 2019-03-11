# "tools" are supposed to be packaged with LDC.
# (in contrast to "utils" which are only used for building / testing)

# Set tool names
set(LDCPRUNECACHE_EXE ldc-prune-cache)
set(LDCPRUNECACHE_EXE ${LDCPRUNECACHE_EXE} PARENT_SCOPE) # needed for correctly populating lit.site.cfg.in
set(LDCPRUNECACHE_EXE_NAME ${PROGRAM_PREFIX}${LDCPRUNECACHE_EXE}${PROGRAM_SUFFIX})
set(LDCPRUNECACHE_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDCPRUNECACHE_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})

function(build_d_tool output_exe compiler_args linker_args compile_deps link_deps)
    set(dflags "${D_COMPILER_FLAGS} ${DDMD_DFLAGS}")
    set(lflags "")
    if(UNIX)
      separate_arguments(dflags UNIX_COMMAND "${dflags}")
      separate_arguments(lflags UNIX_COMMAND "${lflags}")
    else()
      separate_arguments(dflags WINDOWS_COMMAND "${dflags}")
      separate_arguments(lflags WINDOWS_COMMAND "${lflags}")
    endif()
    foreach(f ${linker_args})
        append("-L\"${f}\"" lflags)
    endforeach()
    add_custom_command(
        OUTPUT ${output_exe}
        COMMAND ${D_COMPILER} ${dflags} ${lflags} -I${PROJECT_SOURCE_DIR}/${DDMDFE_PATH} -of${output_exe} ${compiler_args} ${linker_args}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS ${compile_deps} ${link_deps}
    )
endfunction()

#############################################################################
# Build ldc-prune-cache
add_custom_target(${LDCPRUNECACHE_EXE} ALL DEPENDS ${LDCPRUNECACHE_EXE_FULL})
set(LDCPRUNECACHE_D_SRC
    ${PROJECT_SOURCE_DIR}/tools/ldc-prune-cache.d
    ${PROJECT_SOURCE_DIR}/driver/cache_pruning.d
)
build_d_tool(
    "${LDCPRUNECACHE_EXE_FULL}"
    "${LDCPRUNECACHE_D_SRC}"
    ""
    "${LDCPRUNECACHE_D_SRC}"
    ""
)
install(PROGRAMS ${LDCPRUNECACHE_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#############################################################################
# Build ldc-profdata for converting profile data formats (source version depends on LLVM version)
set(LDCPROFDATA_SRC ldc-profdata/llvm-profdata-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LDCPROFDATA_SRC})
    add_executable(ldc-profdata ${LDCPROFDATA_SRC})
    set_target_properties(
        ldc-profdata PROPERTIES
        OUTPUT_NAME "${LDCPROFDATA_EXE_NAME}"
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS}"
    )
    target_link_libraries(ldc-profdata  ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
    install(TARGETS ${LDCPROFDATA_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
