if(LDC_RUNTIME_COMPILE)
#    find_package(LLVM 3.7 REQUIRED support core irreader executionengine passes target nativecodegen)
    file(GLOB LDC_JITRT_D ${JITRT_DIR}/d/ldc/*.d)

    # Choose the correct subfolder depending on the LLVM version
    file(GLOB LDC_JITRT_CXX ${JITRT_DIR}/cpp/*.cpp)
    file(GLOB LDC_JITRT_SO_CXX ${JITRT_DIR}/cpp-so/*.cpp)

    # Set compiler-dependent flags
    if(MSVC)
        # Omit Default Library Name from the library, so it will work with both release and debug builds
        set(JITRT_EXTRA_FLAGS "/Zl")

    else()
        set(JITRT_EXTRA_FLAGS "-fPIC -O3 -std=c++11")
    endif()

    # LLVM_CXXFLAGS may contain -Wcovered-switch-default and -fcolor-diagnostics
    # which are clang-only options
    if(CMAKE_COMPILER_IS_GNUCXX)
        string(REPLACE "-Wcovered-switch-default " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
        string(REPLACE "-fcolor-diagnostics " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    endif()
    # LLVM_CXXFLAGS may contain -Wno-maybe-uninitialized
    # which is gcc-only options
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        string(REPLACE "-Wno-maybe-uninitialized " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    endif()

    # Sets up the targets for building the D-source jit-rt object files,
    # appending the names of the (bitcode) files to link into the library to
    # outlist_o (outlist_bc).
    macro(compile_jit_rt_D d_flags lib_suffix path_suffix all_at_once outlist_o outlist_bc)
        get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
        dc("${LDC_JITRT_D}"
           "${JITRT_DIR}/d"
           "${d_flags}"
           "${PROJECT_BINARY_DIR}/objects${target_suffix}"
           "${all_at_once}"
           ${outlist_o}
           ${outlist_bc}
        )
    endmacro()

    macro(build_jit_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
        get_target_suffix("" "${path_suffix}" target_suffix)

        set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

        add_library(ldc-jit-rt-so${target_suffix} SHARED ${LDC_JITRT_SO_CXX})
        set_target_properties(
            ldc-jit-rt-so${target_suffix} PROPERTIES
            OUTPUT_NAME                 ldc-jit
            VERSION                     ${LDC_VERSION}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags} ${LDC_CXXFLAGS} ${LLVM_CXXFLAGS} ${JITRT_EXTRA_FLAGS}"
            LINK_FLAGS                  "${ld_flags} ${JITRT_EXTRA_LDFLAGS}"
            )

        target_link_libraries(ldc-jit-rt-so${target_suffix} ${LLVM_LIBRARIES})

        set(jitrt_d_o "")
        set(jitrt_d_bc "")
        compile_jit_rt_D("-enable-runtime-compile;${d_flags}" "" "${path_suffix}" "${COMPILE_ALL_D_FILES_AT_ONCE}" jitrt_d_o jitrt_d_bc)

        add_library(ldc-jit-rt${target_suffix} STATIC ${jitrt_d_o} ${LDC_JITRT_CXX})
        set_target_properties(
            ldc-jit-rt${target_suffix} PROPERTIES
            OUTPUT_NAME                 ldc-jit-rt
            VERSION                     ${LDC_VERSION}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags} ${JITRT_EXTRA_FLAGS}"
            LINK_FLAGS                  "${ld_flags} ${JITRT_EXTRA_LDFLAGS}"
            )

        target_link_libraries(ldc-jit-rt${target_suffix} ldc-jit-rt-so${target_suffix})

        list(APPEND ${outlist_targets} "ldc-jit-rt-so${target_suffix}")
        list(APPEND ${outlist_targets} "ldc-jit-rt${target_suffix}")
    endmacro()

    # Install D interface files
    install(DIRECTORY ${JITRT_DIR}/d/ldc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
else()
    macro(build_jit_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
    endmacro()
endif()
