file(GLOB LDC_COMPRT_D ${RTCOMPILE_DIR}/d/ldc/*.d)

# Choose the correct subfolder depending on the LLVM version
file(GLOB LDC_COMPRT_CXX ${RTCOMPILE_DIR}/cpp/*.cpp)
file(GLOB LDC_COMPRT_SO_CXX ${RTCOMPILE_DIR}/cpp-so/*.cpp)

# Set compiler-dependent flags
if(MSVC)
    # Omit Default Library Name from the library, so it will work with both release and debug builds
    #set(COMPRT_EXTRA_FLAGS "/Zl")

else()
    set(COMPRT_EXTRA_FLAGS "-fPIC -O3")
endif()

# Sets up the targets for building the D-source compile-rt object files,
# appending the names of the (bitcode) files to link into the library to
# outlist_o (outlist_bc).
macro(compile_compile_rt_D d_flags lib_suffix path_suffix outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)

    if(BUILD_SHARED_LIBS)
        set(shared ";-d-version=Shared")
    else()
        set(shared)
    endif()

    foreach(f ${LDC_COMPRT_D})
        dc(
            ${f}
            "${d_flags}${shared}"
            "${RTCOMPILE_DIR}"
            "${target_suffix}"
            ${outlist_o}
            ${outlist_bc}
            )
    endforeach()
endmacro()

macro(build_compile_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
    get_target_suffix("" "${path_suffix}" target_suffix)

    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    add_library(ldc-rtcompile-rt-so${target_suffix} SHARED ${LDC_COMPRT_SO_CXX})
    set_target_properties(
        ldc-rtcompile-rt-so${target_suffix} PROPERTIES
        OUTPUT_NAME                 rtcompile
        VERSION                     ${LDC_VERSION}
        LINKER_LANGUAGE             C
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags} ${LDC_CXXFLAGS} ${LLVM_CXXFLAGS} ${COMPRT_EXTRA_FLAGS} /EHsc"
        LINK_FLAGS                  "${ld_flags} ${COMPRT_EXTRA_LDFLAGS}"
        )

    llvm_map_components_to_libnames(llvm_libs support core irreader executionengine passes target nativecodegen)
    target_link_libraries(ldc-rtcompile-rt-so${target_suffix} ${llvm_libs})

    set(compilert_d_o "")
    set(compilert_d_bc "")
    compile_compile_rt_D("${d_flags}" "" "${path_suffix}" compilert_d_o compilert_d_bc)

    add_library(ldc-rtcompile-rt${target_suffix} STATIC ${compilert_d_o} ${LDC_COMPRT_CXX})
    set_target_properties(
        ldc-rtcompile-rt${target_suffix} PROPERTIES
        OUTPUT_NAME                 ldc-rtcompile-rt
        VERSION                     ${LDC_VERSION}
        LINKER_LANGUAGE             C
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags} ${COMPRT_EXTRA_FLAGS} /Zl"
        LINK_FLAGS                  "${ld_flags} ${COMPRT_EXTRA_LDFLAGS}"
        )

    target_link_libraries(ldc-rtcompile-rt${target_suffix} ldc-rtcompile-rt-so${target_suffix})

    list(APPEND ${outlist_targets} "ldc-rtcompile-rt-so${target_suffix}")
    list(APPEND ${outlist_targets} "ldc-rtcompile-rt${target_suffix}")
endmacro()

# Install D interface files
install(DIRECTORY ${RTCOMPILE_DIR}/d/ldc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
