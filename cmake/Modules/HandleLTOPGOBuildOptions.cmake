# Handles the LDC_BUILD_WITH_LTO build option.
# For example `cmake -DLDC_BUILD_WITH_LTO=thin`.
#
# LTO is enabled for the C++ and D compilers, provided that they accept the `-flto` flag.

# TODO: implement a LDC_BUILD_WITH_PGO build option (or something similar) to generate/use an LDC PGO profile.

set(__LTO_FLAG)
set(LDC_BUILD_WITH_LTO OFF CACHE STRING "Build LDC with LTO. May be specified as Thin or Full to use a particular kind of LTO")
string(TOUPPER "${LDC_BUILD_WITH_LTO}" uppercase_LDC_BUILD_WITH_LTO)
if(uppercase_LDC_BUILD_WITH_LTO STREQUAL "THIN")
    set(__LTO_FLAG "-flto=thin")
elseif(uppercase_LDC_BUILD_WITH_LTO STREQUAL "FULL")
    set(__LTO_FLAG "-flto=full")
elseif(LDC_BUILD_WITH_LTO)
    set(__LTO_FLAG "-flto")
endif()
if(__LTO_FLAG)
    message(STATUS "Building LDC using LTO: ${__LTO_FLAG}")
    check_cxx_compiler_flag(${__LTO_FLAG} CXX_COMPILER_ACCEPTS_FLTO_${uppercase_LDC_BUILD_WITH_LTO})
    if (CXX_COMPILER_ACCEPTS_FLTO_${uppercase_LDC_BUILD_WITH_LTO})
        append(${__LTO_FLAG} CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()

    check_d_source_compiles("void main(){}" D_COMPILER_ACCEPTS_FLTO_${uppercase_LDC_BUILD_WITH_LTO} FLAGS ${__LTO_FLAG})
    if(D_COMPILER_ACCEPTS_FLTO_${uppercase_LDC_BUILD_WITH_LTO})
        append(${__LTO_FLAG} DDMD_DFLAGS)
    endif()

    if(uppercase_LDC_BUILD_WITH_LTO STREQUAL "THIN")
        # On darwin, enable the lto cache.
        if(APPLE)
            append("-Wl,-cache_path_lto,${PROJECT_BINARY_DIR}/lto.cache" CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
        endif()
    endif()
    if(uppercase_LDC_BUILD_WITH_LTO MATCHES "^(THIN|FULL)$")
        if (APPLE)
            # Explicitly use the LTO library that shipped with the host LDC, assuming it is newer than the system-installed lib.
            get_filename_component(HOST_LDC_BINDIR ${D_COMPILER} DIRECTORY)
            if(EXISTS ${HOST_LDC_BINDIR}/../lib/libLTO-ldc.dylib)
                message(STATUS "Using ${HOST_LDC_BINDIR}/../lib/libLTO-ldc.dylib for LTO")
                append("-Wl,-lto_library,${HOST_LDC_BINDIR}/../lib/libLTO-ldc.dylib" CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
            endif()
        endif()
    endif()
endif()
