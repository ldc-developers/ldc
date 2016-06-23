# Add LLVM Profile runtime if LDC is built with PGO enabled
if (LDC_WITH_PGO)
    file(GLOB LDC_PROFRT_D ${PROFILERT_DIR}/d/ldc/*.d)

    # Choose the correct subfolder depending on the LLVM version
    set(PROFILERT_LIBSRC_DIR "${PROFILERT_DIR}/profile-rt-${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR}")
    file(GLOB LDC_PROFRT_C ${PROFILERT_LIBSRC_DIR}/*.c)
    file(GLOB LDC_PROFRT_CXX ${PROFILERT_LIBSRC_DIR}/*.cc)

    # Set compiler-dependent flags
    if(MSVC)
        # Omit Default Library Name from the library, so it will work with both release and debug builds
        set(PROFRT_EXTRA_FLAGS "/Zl")

        # Add library needed for `gethostname`
        set(PROFRT_EXTRA_LDFLAGS "Ws2_32.lib")
    else()
        set(PROFRT_EXTRA_FLAGS "-fPIC -O3")
    endif()

    CHECK_CXX_SOURCE_COMPILES("
    #ifdef _MSC_VER
    #include <Intrin.h> /* Workaround for PR19898. */
    #include <windows.h>
    #endif
    int main() {
    #ifdef _MSC_VER
            volatile LONG val = 1;
            MemoryBarrier();
            InterlockedCompareExchange(&val, 0, 1);
            InterlockedIncrement(&val);
            InterlockedDecrement(&val);
    #else
            volatile unsigned long val = 1;
            __sync_synchronize();
            __sync_val_compare_and_swap(&val, 1, 0);
            __sync_add_and_fetch(&val, 1);
            __sync_sub_and_fetch(&val, 1);
    #endif
            return 0;
          }
    " COMPILER_RT_TARGET_HAS_ATOMICS)
    if(COMPILER_RT_TARGET_HAS_ATOMICS)
     set(PROFRT_EXTRA_FLAGS "${PROFRT_EXTRA_FLAGS} -DCOMPILER_RT_HAS_ATOMICS=1")
    endif()

    CHECK_CXX_SOURCE_COMPILES("
    #if defined(__linux__)
    #include <unistd.h>
    #endif
    #include <fcntl.h>
    int fd;
    int main() {
     struct flock s_flock;

     s_flock.l_type = F_WRLCK;
     fcntl(fd, F_SETLKW, &s_flock);
     return 0;
    }
    " COMPILER_RT_TARGET_HAS_FCNTL_LCK)
    if(COMPILER_RT_TARGET_HAS_FCNTL_LCK)
     set(PROFRT_EXTRA_FLAGS "${PROFRT_EXTRA_FLAGS} -DCOMPILER_RT_HAS_FCNTL_LCK=1")
    endif()

    # Sets up the targets for building the D-source profile-rt object files,
    # appending the names of the (bitcode) files to link into the library to
    # outlist_o (outlist_bc).
    macro(compile_profilert_D d_flags lib_suffix path_suffix outlist_o outlist_bc)
        get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)

        if(BUILD_SHARED_LIBS)
            set(shared ";-d-version=Shared")
        else()
            set(shared)
        endif()

        foreach(f ${LDC_PROFRT_D})
            dc(
                ${f}
                "${d_flags}${shared}"
                "${PROFILERT_DIR}"
                "${target_suffix}"
                ${outlist_o}
                ${outlist_bc}
            )
        endforeach()
    endmacro()

    macro(build_profile_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
        get_target_suffix("" "${path_suffix}" target_suffix)

        set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

        set(profilert_d_o "")
        set(profilert_d_bc "")
        compile_profilert_D("${d_flags}" "" "${path_suffix}" profilert_d_o profilert_d_bc)

        add_library(ldc-profile-rt${target_suffix} STATIC ${profilert_d_o} ${LDC_PROFRT_C} ${LDC_PROFRT_CXX})
        set_target_properties(
            ldc-profile-rt${target_suffix} PROPERTIES
            OUTPUT_NAME                 ldc-profile-rt
            VERSION                     ${LDC_VERSION}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags} ${PROFRT_EXTRA_FLAGS}"
            LINK_FLAGS                  "${ld_flags} ${PROFRT_EXTRA_LDFLAGS}"
        )

        list(APPEND ${outlist_targets} "ldc-profile-rt${target_suffix}")
    endmacro()

    # Install D interface files to profile-rt.
    install(DIRECTORY ${PROFILERT_DIR}/d/ldc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")

else()
    # No profiling supported, define NOP macro
    macro(build_profile_runtime c_flags ld_flags path_suffix outlist_targets)
    endmacro()
endif()