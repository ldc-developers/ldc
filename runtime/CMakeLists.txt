project(runtime)

cmake_minimum_required(VERSION 2.8.5)

#
# Main configuration.
#

set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                       CACHE BOOL    "Build both 32/64 bit runtime libraries")
set(BUILD_BC_LIBS         OFF                                       CACHE BOOL    "Build the runtime as LLVM bitcode libraries")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d         CACHE PATH    "Path to install D modules to")
set(BUILD_SHARED_LIBS     OFF                                       CACHE BOOL    "Whether to build the runtime as a shared library")
set(D_FLAGS               -w                                        CACHE STRING  "Runtime build flags, separated by ;")
set(D_FLAGS_DEBUG         -g;-link-debuglib                         CACHE STRING  "Runtime build flags (debug libraries), separated by ;")
set(D_FLAGS_RELEASE       -O3;-release                              CACHE STRING  "Runtime build flags (release libraries), separated by ;")
if(MSVC)
    set(LINK_WITH_MSVCRT  OFF                                       CACHE BOOL    "Link with MSVCRT.lib (DLL) instead of LIBCMT.lib (static)")
endif()

set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^8$") ## if it's 64-bit OS
    set(HOST_BITNESS 64)
    set(MULTILIB_SUFFIX 32)
else()
    set(HOST_BITNESS 32)
    set(MULTILIB_SUFFIX 64)
endif()

if(BUILD_SHARED_LIBS)
    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND
       NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD" AND
       NOT APPLE)
        message(FATAL_ERROR "Shared libraries (BUILD_SHARED_LIBS) are only supported on Linux, macOS and FreeBSD for the time being.")
    endif()

    list(APPEND D_FLAGS -relocation-model=pic)
    set(D_LIBRARY_TYPE SHARED)
else()
    set(D_LIBRARY_TYPE STATIC)
endif()

get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)
set(RUNTIME_DIR ${PROJECT_SOURCE_DIR}/druntime CACHE PATH "druntime root directory")
set(PHOBOS2_DIR ${PROJECT_SOURCE_DIR}/phobos CACHE PATH "Phobos root directory")
set(PROFILERT_DIR ${PROJECT_SOURCE_DIR}/profile-rt CACHE PATH "profile-rt root directory")

#
# Gather source files.
#

# druntime D parts
file(GLOB_RECURSE DRUNTIME_D ${RUNTIME_DIR}/src/*.d)
list(REMOVE_ITEM DRUNTIME_D ${RUNTIME_DIR}/src/test_runner.d)
# remove unsupported etc/linux/memoryerror.d (see issue #1915)
list(REMOVE_ITEM DRUNTIME_D ${RUNTIME_DIR}/src/etc/linux/memoryerror.d)
# FIXME: Remove all modules in core/stdcpp/ due to alpha quality.
#        See PR #1917; partly fixed upstream.
file(GLOB_RECURSE DRUNTIME_D_STDCPP ${RUNTIME_DIR}/src/core/stdcpp/*.d)
list(REMOVE_ITEM DRUNTIME_D ${DRUNTIME_D_STDCPP})
# remove all modules in gcstub/
file(GLOB_RECURSE DRUNTIME_D_GCSTUB ${RUNTIME_DIR}/src/gcstub/*.d)
list(REMOVE_ITEM DRUNTIME_D ${DRUNTIME_D_GCSTUB})
# remove some modules in rt/
list(REMOVE_ITEM DRUNTIME_D
    ${RUNTIME_DIR}/src/rt/alloca.d
    ${RUNTIME_DIR}/src/rt/deh.d
    ${RUNTIME_DIR}/src/rt/deh_win32.d
    ${RUNTIME_DIR}/src/rt/deh_win64_posix.d
    ${RUNTIME_DIR}/src/rt/dwarfeh.d
    ${RUNTIME_DIR}/src/rt/llmath.d
    ${RUNTIME_DIR}/src/rt/trace.d
)
# only include core/sys/ modules matching the platform
file(GLOB_RECURSE DRUNTIME_D_BIONIC ${RUNTIME_DIR}/src/core/sys/bionic/*.d)
file(GLOB_RECURSE DRUNTIME_D_DARWIN ${RUNTIME_DIR}/src/core/sys/darwin/*.d)
file(GLOB_RECURSE DRUNTIME_D_FREEBSD ${RUNTIME_DIR}/src/core/sys/freebsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_LINUX ${RUNTIME_DIR}/src/core/sys/linux/*.d)
file(GLOB_RECURSE DRUNTIME_D_OPENBSD ${RUNTIME_DIR}/src/core/sys/openbsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_OSX ${RUNTIME_DIR}/src/core/sys/osx/*.d)
file(GLOB_RECURSE DRUNTIME_D_POSIX ${RUNTIME_DIR}/src/core/sys/posix/*.d)
file(GLOB_RECURSE DRUNTIME_D_SOLARIS ${RUNTIME_DIR}/src/core/sys/solaris/*.d)
file(GLOB_RECURSE DRUNTIME_D_WINDOWS ${RUNTIME_DIR}/src/core/sys/windows/*.d)
list(REMOVE_ITEM DRUNTIME_D
    ${DRUNTIME_D_BIONIC}  ${DRUNTIME_D_DARWIN} ${DRUNTIME_D_FREEBSD} ${DRUNTIME_D_LINUX}
    ${DRUNTIME_D_OPENBSD} ${DRUNTIME_D_OSX}    ${DRUNTIME_D_POSIX}   ${DRUNTIME_D_SOLARIS}
    ${DRUNTIME_D_WINDOWS}
)
if(UNIX)
    list(APPEND DRUNTIME_D ${DRUNTIME_D_POSIX})
    if(APPLE)
        list(APPEND DRUNTIME_D ${DRUNTIME_D_DARWIN})
        list(APPEND DRUNTIME_D ${DRUNTIME_D_OSX})
    elseif(${CMAKE_SYSTEM} MATCHES "FreeBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_FREEBSD})
    elseif(${CMAKE_SYSTEM} MATCHES "Linux")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_LINUX})
        list(APPEND DRUNTIME_D ${DRUNTIME_D_BIONIC})
    elseif(${CMAKE_SYSTEM} MATCHES "OpenBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_OPENBSD})
    elseif(${CMAKE_SYSTEM} MATCHES "SunOS")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_SOLARIS})
    endif()
elseif(WIN32)
    list(APPEND DRUNTIME_D ${DRUNTIME_D_WINDOWS})
endif()

# druntime C parts
file(GLOB_RECURSE DRUNTIME_C ${RUNTIME_DIR}/src/*.c)
list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/dylib_fixes.c)
if(NOT APPLE)
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/osx_tls.c)
endif()
if(NOT MSVC)
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/msvc.c ${RUNTIME_DIR}/src/rt/msvc_math.c)
endif()

# druntime ASM parts
set(DRUNTIME_ASM)
if(UNIX)
    list(APPEND DRUNTIME_ASM ${RUNTIME_DIR}/src/core/threadasm.S ${RUNTIME_DIR}/src/ldc/eh_asm.S)
endif()

if(PHOBOS2_DIR)
    # Phobos D parts
    file(GLOB_RECURSE PHOBOS2_D ${PHOBOS2_DIR}/*.d)
    # remove top-level modules index.d and unittest.d
    list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_DIR}/index.d ${PHOBOS2_DIR}/unittest.d)
    # only include std/c/ modules matching the platform
    file(GLOB_RECURSE PHOBOS2_D_FREEBSD ${PHOBOS2_DIR}/std/c/freebsd/*.d)
    file(GLOB_RECURSE PHOBOS2_D_LINUX ${PHOBOS2_DIR}/std/c/linux/*.d)
    file(GLOB_RECURSE PHOBOS2_D_OSX ${PHOBOS2_DIR}/std/c/osx/*.d)
    file(GLOB_RECURSE PHOBOS2_D_WINDOWS ${PHOBOS2_DIR}/std/c/windows/*.d)
    list(REMOVE_ITEM PHOBOS2_D
        ${PHOBOS2_D_FREEBSD} ${PHOBOS2_D_LINUX} ${PHOBOS2_D_OSX} ${PHOBOS2_D_WINDOWS}
    )
    if(UNIX)
        if(APPLE)
            list(APPEND PHOBOS2_D ${PHOBOS2_D_OSX})
        elseif(${CMAKE_SYSTEM} MATCHES "FreeBSD")
            list(APPEND PHOBOS2_D ${PHOBOS2_D_FREEBSD})
        elseif(${CMAKE_SYSTEM} MATCHES "Linux")
            list(APPEND PHOBOS2_D ${PHOBOS2_D_LINUX})
        endif()
    elseif(WIN32)
        list(APPEND PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()
    # only include std/windows/ modules on Windows
    if(NOT WIN32)
        file(GLOB_RECURSE PHOBOS2_D_WINDOWS ${PHOBOS2_DIR}/std/windows/*.d)
        list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()

    # Phobos C parts
    file(GLOB_RECURSE PHOBOS2_C ${PHOBOS2_DIR}/*.c)
    # remove some obsolete zlib modules
    list(REMOVE_ITEM PHOBOS2_C ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c ${PHOBOS2_DIR}/etc/c/zlib/example.c)

    set(CONFIG_NAME ${LDC_EXE}_phobos)
else()
    set(CONFIG_NAME ${LDC_EXE})
endif()

# should only be necessary if run independently from ldc cmake project
if(NOT LDC_EXE)
    find_program(LDC_EXE ldc2 ${PROJECT_BINARY_DIR}/../bin DOC "path to ldc binary")
    if(NOT LDC_EXE)
        message(SEND_ERROR "ldc not found")
    endif()
endif()

#
# Create configuration files.
#

# Add extra paths on Linux and disable linker arch mismatch warnings (like
# DMD and GDC do). OS X doesn't need extra configuration due to the use of
# fat binaries. Other Posixen might need to be added here.
if((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD"))
    if(MULTILIB)
        set(MULTILIB_ADDITIONAL_PATH         "\n        \"-L-L${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
        set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"-L-L${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
    endif()

    if(BUILD_SHARED_LIBS)
        set(SHARED_LIBS_RPATH "\n        \"-L-rpath=${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}\",")
    endif()
endif()

configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
# Prepare the config files we are going to install later in bin.
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)

#
# druntime/Phobos compilation helpers.
#

set(GCCBUILTINS "")
function(gen_gccbuiltins name)
  set(module "${PROJECT_BINARY_DIR}/gccbuiltins_${name}.di")
  if (GCCBUILTINS STREQUAL "")
    set(GCCBUILTINS "${module}" PARENT_SCOPE)
  else()
    set(GCCBUILTINS "${GCCBUILTINS};${module}" PARENT_SCOPE)
  endif()
  add_custom_command(
      OUTPUT ${module}
      COMMAND gen_gccbuiltins ${module} "${name}"
  )
endfunction()

set(target_arch "AArch64;ARM;Mips;PowerPC;SystemZ;X86")
set(target_name "aarch64;arm;mips;ppc;s390;x86")

foreach(target ${LLVM_TARGETS_TO_BUILD})
  list(FIND target_arch ${target} idx)
  if(idx GREATER -1)
    list(GET target_name ${idx} name)
    gen_gccbuiltins(${name})
  endif()
endforeach()

# Always build zlib and other C parts of the runtime in release mode, regardless
# of what the user chose for LDC itself.
# 1) Set up CMAKE_C_FLAGS_RELEASE
if(MSVC)
    if(NOT LINK_WITH_MSVCRT)
        string(REGEX REPLACE "(^| )[/-]MD( |$)" "\\2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        if(NOT CMAKE_C_FLAGS_RELEASE MATCHES "(^| )[/-]MT( |$)")
            append("/MT" CMAKE_C_FLAGS_RELEASE)
        endif()
    else()
        string(REGEX REPLACE "(^| )[/-]MT( |$)" "\\2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        if(NOT CMAKE_C_FLAGS_RELEASE MATCHES "(^| )[/-]MD( |$)")
            append("/MD" CMAKE_C_FLAGS_RELEASE)
        endif()
    endif()
    # warning C4996: zlib uses 'deprecated' POSIX names
    append("/wd4996" CMAKE_C_FLAGS_RELEASE)
endif()
# 2) Set all other CMAKE_C_FLAGS variants to CMAKE_C_FLAGS_RELEASE
set(variables
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
)
foreach(variable ${variables})
    set(${variable} "${CMAKE_C_FLAGS_RELEASE}")
endforeach()

# Compiles the given D module into an object file, and if enabled, a bitcode
# file. The output is written to a path based on output_dir. The paths of the
# output files are appended to outlist_o and outlist_bc, respectively.
macro(dc input_d d_flags output_dir output_suffix outlist_o outlist_bc)
    file(RELATIVE_PATH output ${output_dir} ${input_d})

    get_filename_component(name ${output} NAME_WE)
    get_filename_component(path ${output} PATH)
    if("${path}" STREQUAL "")
        set(output_root ${name})
    else()
        set(output_root ${path}/${name})
    endif()

    set(output_o  ${PROJECT_BINARY_DIR}/${output_root}${output_suffix}${CMAKE_C_OUTPUT_EXTENSION})
    set(output_bc ${PROJECT_BINARY_DIR}/${output_root}${output_suffix}.bc)
    list(APPEND ${outlist_o} ${output_o})
    if(BUILD_BC_LIBS)
        list(APPEND ${outlist_bc} ${output_bc})
    endif()

    # Compile
    if(BUILD_BC_LIBS)
        set(outfiles ${output_o} ${output_bc})
        set(dc_flags --output-o --output-bc)
    else()
        set(outfiles ${output_o})
        set(dc_flags --output-o)
    endif()

    add_custom_command(
        OUTPUT
            ${outfiles}
        COMMAND ${LDC_EXE_FULL} ${dc_flags} -c -I${RUNTIME_DIR}/src -I${RUNTIME_DIR}/src/gc ${input_d} -of${output_o} ${d_flags}
        WORKING_DIRECTORY ${PROJECT_PARENT_DIR}
        DEPENDS ${input_d}
                ${LDC_EXE}
                ${LDC_EXE_FULL}
                ${GCCBUILTINS}
                ${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in
    )
endmacro()

# Sets target_suffix to a purely cosmetical suffix for the CMake target names
# from the given suffixes on the library name and the target path. We could use
# any string that resolves the ambiguities between the different variants.
macro(get_target_suffix lib_suffix path_suffix target_suffix)
    set(${target_suffix} "")
    if(NOT "${lib_suffix}" STREQUAL "")
        set(${target_suffix} "${lib_suffix}")
    endif()

    # If LIB_SUFFIX is set there is always a suffix; leave it off for simplicity.
    if(NOT "${path_suffix}" STREQUAL "" AND NOT "${path_suffix}" STREQUAL "${LIB_SUFFIX}")
        set(${target_suffix} "${${target_suffix}}_${path_suffix}")
    endif()
endmacro()

# Sets up the targets for building the individual druntime object files,
# appending the names of the (bitcode) files to link into the library to
# outlist_o (outlist_bc).
macro(compile_druntime d_flags lib_suffix path_suffix outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)

    if(BUILD_SHARED_LIBS)
        set(shared ";-d-version=Shared")
    else()
        set(shared)
    endif()

    foreach(f ${DRUNTIME_D})
        dc(
            ${f}
            "${d_flags}${shared}"
            "${RUNTIME_DIR}"
            "${target_suffix}"
            ${outlist_o}
            ${outlist_bc}
        )
    endforeach()
endmacro()

# Sets up the targets for building the individual Phobos object files,
# appending the names of the (bitcode) files to link into the library to
# outlist_o (outlist_bc).
macro(compile_phobos2 d_flags lib_suffix path_suffix outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    foreach(f ${PHOBOS2_D})
        dc(
            ${f}
            "${d_flags};-I${PHOBOS2_DIR}"
            ${PHOBOS2_DIR}
            "${target_suffix}"
            ${outlist_o}
            ${outlist_bc}
        )
    endforeach()
endmacro()

# Sets the common properties for all library targets.
function(set_common_library_properties target)
    set_target_properties(${target} PROPERTIES
        VERSION ${DMDFE_VERSION}
        SOVERSION ${DMDFE_PATCH_VERSION}
        LINKER_LANGUAGE C
    )

    # ldc2 defaults to position-independent code on Linux to match the implicit
    # linker default on Ubuntu 16.10 and above. As we might be building on an
    # older system (e.g. binary packages), we need to make sure the C parts are
    # built as PIC as well.
    if(${CMAKE_SYSTEM} MATCHES "Linux")
        set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

# Builds a copy of druntime/Phobos from the source files gathered above. The
# names of the added library targets are appended to outlist_targets.
macro(build_runtime d_flags c_flags ld_flags lib_suffix path_suffix outlist_targets)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    set(druntime_o "")
    set(druntime_bc "")
    compile_druntime("${d_flags}" "${lib_suffix}" "${path_suffix}" druntime_o druntime_bc)

    add_library(druntime-ldc${target_suffix} ${D_LIBRARY_TYPE}
        ${druntime_o} ${DRUNTIME_C} ${DRUNTIME_ASM})
    set_target_properties(
        druntime-ldc${target_suffix} PROPERTIES
        OUTPUT_NAME                 druntime-ldc${lib_suffix}
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${ld_flags}"
    )
    set_common_library_properties(druntime-ldc${target_suffix})

    # When building a shared library, we need to link in all the default
    # libraries otherwise implicitly added by LDC to make it loadable from
    # C executables.

    if(BUILD_SHARED_LIBS)
        if(${CMAKE_SYSTEM} MATCHES "Linux")
            set(dso_system_libs "m" "pthread" "rt" "dl")
        else()
            set(dso_system_libs "m" "pthread")
        endif()
        target_link_libraries(druntime-ldc${target_suffix} ${dso_system_libs})
    endif()

    list(APPEND ${outlist_targets} druntime-ldc${target_suffix})

    if(PHOBOS2_DIR)
        set(phobos2_o "")
        set(phobos2_bc "")
        compile_phobos2("${d_flags}" "${lib_suffix}" "${path_suffix}" phobos2_o phobos2_bc)

        add_library(phobos2-ldc${target_suffix} ${D_LIBRARY_TYPE}
            ${phobos2_o} ${PHOBOS2_C})
        set_target_properties(
            phobos2-ldc${target_suffix} PROPERTIES
            OUTPUT_NAME                 phobos2-ldc${lib_suffix}
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags}"
            LINK_FLAGS                  "${ld_flags}"
        )
        set_common_library_properties(phobos2-ldc${target_suffix})

        if(BUILD_SHARED_LIBS)
            # TODO: As for druntime, adapt once shared libraries are supported
            # on more operating systems.
            target_link_libraries(phobos2-ldc${target_suffix}
                druntime-ldc${target_suffix} ${dso_system_libs})
        endif()

        list(APPEND ${outlist_targets} "phobos2-ldc${target_suffix}")
    endif()

    if(BUILD_BC_LIBS)
        find_program(LLVM_AR_EXE llvm-ar
            HINTS ${LLVM_ROOT_DIR}/bin
            DOC "path to llvm-ar tool"
        )
        if(NOT LLVM_AR_EXE)
            message(SEND_ERROR "llvm-ar not found")
        endif()

        set(bclibs
            ${output_path}/libdruntime-ldc${lib_suffix}-bc.a
            ${output_path}/libphobos2-ldc${lib_suffix}-bc.a
        )
        add_custom_command(
            OUTPUT ${bclibs}
            COMMAND ${LLVM_AR_EXE} rs libdruntime-ldc${lib_suffix}-bc.a ${druntime_bc}
            COMMAND ${LLVM_AR_EXE} rs libphobos2-ldc${lib_suffix}-bc.a ${phobos2_bc}
            WORKING_DIRECTORY ${output_path}
            DEPENDS
                ${druntime_bc}
                ${phobos2_bc}
            VERBATIM
        )

        add_custom_target(bitcode-libraries${target_suffix} ALL DEPENDS ${bclibs})
    endif()
endmacro()

# Builds both a debug and a release copy of druntime/Phobos.
macro(build_runtime_variants d_flags c_flags ld_flags path_suffix outlist_targets)
    build_runtime(
        "${d_flags};${D_FLAGS};${D_FLAGS_RELEASE}"
        "${c_flags}"
        "${ld_flags}"
        ""
        "${path_suffix}"
        ${outlist_targets}
    )
    build_runtime(
        "${d_flags};${D_FLAGS};${D_FLAGS_DEBUG}"
        "${c_flags}"
        "${ld_flags}"
        "-debug"
        "${path_suffix}"
        ${outlist_targets}
    )
    build_profile_runtime ("${d_flags}" "${c_flags}" "${ld_flags}" "${path_suffix}" ${outlist_targets})
endmacro()

# Setup the build of profile-rt
include(profile-rt/DefineBuildProfileRT.cmake)

#
# Set up build targets.
#

set(RT_CFLAGS "")

# This is a bit of a mess as we need to join the two libraries together on
# OS X before installing them. After this has run, LIBS_TO_INSTALL contains
# a list of library "base names" to install (i.e. without the multilib suffix,
# if any).
set(LIBS_TO_INSTALL)
if(BUILD_SHARED_LIBS)
    set(OSX_LIBEXT "dylib")
else()
    set(OSX_LIBEXT "a")
endif()
if(MULTILIB)
    if(APPLE)
        # On OS X, build a "fat" library.

        # Some suffix for the target/file names of the host-native arch so
        # that they don't collide with the final combined ones.
        set(hostsuffix "${LIB_SUFFIX}${HOST_BITNESS}")

        set(hosttargets)
        build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${hostsuffix}" hosttargets)
        #build_profile_runtime ("" "${RT_CFLAGS}" "${LD_FLAGS}" "${hostsuffix}" hosttargets)

        set(multitargets)
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" multitargets)
        #build_profile_runtime ("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" multitargets)

        foreach(targetname ${hosttargets})
            string(REPLACE "_${hostsuffix}" "" t ${targetname})

            add_custom_command(
                OUTPUT ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${t}.${OSX_LIBEXT}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}
                COMMAND "lipo"
                ARGS ${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}/lib${t}.${OSX_LIBEXT} ${CMAKE_BINARY_DIR}/lib${hostsuffix}/lib${t}.${OSX_LIBEXT} -create -output ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${t}.${OSX_LIBEXT}
                DEPENDS ${hosttargets} ${multitargets}
            )

            add_custom_target(${t} ALL DEPENDS ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${t}.${OSX_LIBEXT})
            list(APPEND LIBS_TO_INSTALL ${t})
        endforeach()
    else()
        build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
        #build_profile_runtime ("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" dummy)
        #build_profile_runtime ("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" dummy)
    endif()
else()
    build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
    #build_profile_runtime ("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
endif()

#
# Install target.
#

set(DRUNTIME_PACKAGES core etc ldc)

install(FILES ${RUNTIME_DIR}/src/object.d DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)
foreach(p ${DRUNTIME_PACKAGES})
    install(DIRECTORY ${RUNTIME_DIR}/src/${p} DESTINATION ${INCLUDE_INSTALL_DIR})
endforeach()
if(PHOBOS2_DIR)
    install(DIRECTORY ${PHOBOS2_DIR}/std DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
    install(DIRECTORY ${PHOBOS2_DIR}/etc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
endif()
install(FILES ${GCCBUILTINS} DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)

foreach(libname ${LIBS_TO_INSTALL})
    if(APPLE)
        install(
            FILES ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${libname}.${OSX_LIBEXT}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}
        )
    else()
        install(
            TARGETS ${libname}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}
        )
        if(MULTILIB)
            install(
                TARGETS ${libname}_${MULTILIB_SUFFIX}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}
            )
        endif()
    endif()
endforeach()


#
# Test targets.
#

# Build the "test runner" executables containing the druntime and Phobos unit
# tests. They are invoked with the modules to test later.
# We just build another copy of the two libraries with -unittest enabled and
# link the test runners against those. Some linker command-line magic is
# required to make sure all objects are pulled in.

macro(append_flags_to_pull_in_all_objects lib name_suffix target_suffix output_flags)
    if(BUILD_SHARED_LIBS)
        if(MSVC OR APPLE)
            list(APPEND ${output_flags} "-L$<TARGET_LINKER_FILE:${lib}${target_suffix}>")
        else()
            list(APPEND ${output_flags} -L--no-as-needed -L-l${lib}${name_suffix} -L--as-needed)
        endif()
    else()
        if(MSVC)
            # the MS linker supports /WHOLEARCHIVE since VS 2015 Update 2
            list(APPEND ${output_flags} -L/WHOLEARCHIVE:${lib}${name_suffix})
        elseif(APPLE)
            list(APPEND ${output_flags} "-L-Wl,-force_load,$<TARGET_LINKER_FILE:${lib}${target_suffix}>")
        else()
            list(APPEND ${output_flags} -L--whole-archive -L-l${lib}${name_suffix} -L--no-whole-archive)
        endif()
    endif()
endmacro()

macro(build_test_runner name_suffix path_suffix d_flags c_flags)
    set(flags "${D_FLAGS};${d_flags};-unittest")

    set(unittest_libs "")
    build_runtime(
        "${flags}"
        "${RT_CFLAGS} ${c_flags}"
        "${LD_FLAGS} ${c_flags}"
        "-unittest${name_suffix}"
        "${path_suffix}"
        unittest_libs
    )

    # Only build the unittest libraries when running the tests. Unfortunately,
    # I couldn't find an easier way to make a test depend on a CMake target than
    # just invoking the build command through the CMake executable.
    set_target_properties(${unittest_libs} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)
    foreach(l ${unittest_libs})
        add_test(build-${l} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${l})
    endforeach()

    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)

    set(tmpflags "${flags}")
    append_flags_to_pull_in_all_objects("druntime-ldc-unittest" "${name_suffix}" "${target_suffix}" tmpflags)
    add_test(NAME build-druntime-test-runner${target_suffix}
        COMMAND ${LDC_EXE_FULL}
            -of${PROJECT_BINARY_DIR}/druntime-test-runner${target_suffix}${CMAKE_EXECUTABLE_SUFFIX}
            -defaultlib= -debuglib=
            ${tmpflags} ${RUNTIME_DIR}/src/test_runner.d
    )
    set_tests_properties(build-druntime-test-runner${target_suffix} PROPERTIES
        DEPENDS build-druntime-ldc-unittest${target_suffix})

    if(PHOBOS2_DIR)
        set(tmpflags "${flags}")
        append_flags_to_pull_in_all_objects("phobos2-ldc-unittest" "${name_suffix}" "${target_suffix}" tmpflags)
        set(libarg "druntime-ldc-unittest${name_suffix}")
        add_test(NAME build-phobos2-test-runner${target_suffix}
            COMMAND ${LDC_EXE_FULL}
                -of${PROJECT_BINARY_DIR}/phobos2-test-runner${target_suffix}${CMAKE_EXECUTABLE_SUFFIX}
                -defaultlib=${libarg} -debuglib=${libarg}
                ${tmpflags} ${RUNTIME_DIR}/src/test_runner.d
        )
        set_tests_properties(build-phobos2-test-runner${target_suffix} PROPERTIES
            DEPENDS "build-druntime-ldc-unittest${target_suffix};build-phobos2-ldc-unittest${target_suffix}"
        )
    endif()
endmacro()

build_test_runner("" "${LIB_SUFFIX}" "${D_FLAGS_RELEASE}" "")
build_test_runner("-debug" "${LIB_SUFFIX}" "${D_FLAGS_DEBUG}" "")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    build_test_runner("" "${MULTILIB_SUFFIX}" "${D_FLAGS_RELEASE};-m32" "-m32")
    build_test_runner("-debug" "${MULTILIB_SUFFIX}" "${D_FLAGS_DEBUG};-m32" "-m32")
endif()

# Add the druntime/Phobos test runner invocations for all the different modules.

macro(file_to_module_name file_name out_module_name)
    string(REPLACE ${PROJECT_SOURCE_DIR}/ "" stripped ${file_name})
    string(REPLACE "druntime/src/" "" stripped ${stripped})
    string(REPLACE "phobos/" "" stripped ${stripped})
    string(REPLACE ".d" "" stripped ${stripped})
    string(REPLACE "/" "." module ${stripped})

    # The logical module name for package.d files doesn't include the
    # trailing .package part.
    string(REPLACE ".package" "" module ${module})

    # rt.invariant doesn't have a module declaration, presumably because
    # invariant is a keyword.
    string(REPLACE "rt.invariant" "invariant" ${out_module_name} ${module})
endmacro()

function(add_tests d_files runner target_suffix)
    foreach(file ${d_files})
        file_to_module_name(${file} module)
        add_test(NAME "${module}${target_suffix}"
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
            COMMAND ${runner}-test-runner${target_suffix} ${module}
        )
        set_tests_properties("${module}${target_suffix}" PROPERTIES
            DEPENDS build-${runner}-test-runner${target_suffix}
        )
    endforeach()
endfunction()
function(add_runtime_tests name_suffix path_suffix)
    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)
    add_tests("${DRUNTIME_D}" "druntime" "${target_suffix}")
    if(PHOBOS2_DIR)
        add_tests("${PHOBOS2_D}" "phobos2" "${target_suffix}")
    endif()
endfunction()

add_runtime_tests("" "${LIB_SUFFIX}")
add_runtime_tests("-debug" "${LIB_SUFFIX}")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    add_runtime_tests("" "${MULTILIB_SUFFIX}")
    add_runtime_tests("-debug" "${MULTILIB_SUFFIX}")
endif()

# Add the standalone druntime tests.
# TODO: Add test/excetions and test/init_fini.
if(BUILD_SHARED_LIBS)
    set(druntime_path "$<TARGET_FILE:druntime-ldc>")
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-shared)

    add_test(NAME clean-druntime-test-shared
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})

    if(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
        set(linkflags "")
    else()
        set(linkflags "LINKDL=-L-ldl")
    endif()

    add_test(NAME druntime-test-shared
        COMMAND make -C ${PROJECT_SOURCE_DIR}/druntime/test/shared
            ROOT=${outdir} DMD=${LDMD_EXE_FULL} MODEL=default DRUNTIMESO=${druntime_path}
            CFLAGS=-Wall\ -Wl,-rpath,${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX} ${linkflags}
    )

    set_tests_properties(druntime-test-shared PROPERTIES DEPENDS clean-druntime-test-shared)
endif()
