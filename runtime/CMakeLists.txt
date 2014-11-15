project(runtime)

cmake_minimum_required(VERSION 2.6)

#
# Main configuration.
#

set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                       CACHE BOOL    "Build both 32/64 bit runtime libraries")
set(BUILD_BC_LIBS         OFF                                       CACHE BOOL    "Build the runtime as LLVM bitcode libraries")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d         CACHE PATH    "Path to install D modules to")
set(BUILD_SHARED_LIBS     OFF                                       CACHE BOOL    "Whether to build the runtime as a shared library")
set(D_FLAGS               -w;-d                                     CACHE STRING  "Runtime build flags, separated by ;")
set(D_FLAGS_DEBUG         -g                                        CACHE STRING  "Runtime build flags (debug libraries), separated by ;")
set(D_FLAGS_RELEASE       -O3;-release                              CACHE STRING  "Runtime build flags (release libraries), separated by ;")
if(MSVC)
    set(LINK_WITH_MSVCRT  OFF                                       CACHE BOOL    "Link with MSVCRT.LIB instead of LIBCMT.LIB")
endif()

# Note: In the below building helpers, this is more fittingly called
# ${path_suffix}. ${lib_suffix} refers to the library file suffix there
# (e.g. -debug).
set(LIB_SUFFIX "" CACHE STRING "'64' to install libraries into ${PREFIX}/lib64")

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
    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        message(FATAL_ERROR "Shared libraries (BUILD_SHARED_LIBS) are only supported on Linux for the time being.")
    endif()

    list(APPEND D_FLAGS -relocation-model=pic)
    set(D_LIBRARY_TYPE SHARED)
else()
    set(D_LIBRARY_TYPE STATIC)
    set(CXX_COMPILE_FLAGS " ")
endif()

get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)
set(RUNTIME_DIR ${PROJECT_SOURCE_DIR}/druntime CACHE PATH "druntime root directory")
set(PHOBOS2_DIR ${PROJECT_SOURCE_DIR}/phobos CACHE PATH "Phobos root directory")

#
# Gather source files.
#

file(GLOB CORE_D ${RUNTIME_DIR}/src/core/*.d)
file(GLOB_RECURSE CORE_D_INTERNAL ${RUNTIME_DIR}/src/core/internal/*.d)
file(GLOB CORE_D_SYNC ${RUNTIME_DIR}/src/core/sync/*.d)
file(GLOB CORE_D_STDC ${RUNTIME_DIR}/src/core/stdc/*.d)
file(GLOB_RECURSE GC_D ${RUNTIME_DIR}/src/gc/*.d)
file(GLOB_RECURSE DCRT_D ${RUNTIME_DIR}/src/rt/*.d)
file(GLOB_RECURSE LDC_D ${RUNTIME_DIR}/src/ldc/*.d)
list(REMOVE_ITEM DCRT_D
    ${RUNTIME_DIR}/src/rt/alloca.d
    ${RUNTIME_DIR}/src/rt/deh.d
    ${RUNTIME_DIR}/src/rt/deh_win32.d
    ${RUNTIME_DIR}/src/rt/deh_win64_posix.d
    ${RUNTIME_DIR}/src/rt/llmath.d
    ${RUNTIME_DIR}/src/rt/trace.d
)
file(GLOB DCRT_C ${RUNTIME_DIR}/src/rt/*.c)
list(REMOVE_ITEM DCRT_C ${RUNTIME_DIR}/src/rt/deh.c ${RUNTIME_DIR}/src/rt/dylib_fixes.c)
if(APPLE)
    list(APPEND DCRT_C ${RUNTIME_DIR}/src/ldc/osx_tls.c)
endif()
file(GLOB_RECURSE CORE_D_UNIX ${RUNTIME_DIR}/src/core/sys/posix/*.d)
file(GLOB_RECURSE CORE_D_FREEBSD ${RUNTIME_DIR}/src/core/sys/freebsd/*.d)
file(GLOB_RECURSE CORE_D_LINUX ${RUNTIME_DIR}/src/core/sys/linux/*.d)
file(GLOB_RECURSE CORE_D_OSX ${RUNTIME_DIR}/src/core/sys/osx/*.d)
file(GLOB_RECURSE CORE_D_WIN ${RUNTIME_DIR}/src/core/sys/windows/*.d)
set(CORE_D_SYS)
set(DCRT_ASM)
if(UNIX)
    list(APPEND CORE_D_SYS ${CORE_D_UNIX})
    if(${CMAKE_SYSTEM} MATCHES "FreeBSD")
        list(APPEND CORE_D_SYS ${CORE_D_FREEBSD})
    endif()
    if(${CMAKE_SYSTEM} MATCHES "Linux")
        list(APPEND CORE_D_SYS ${CORE_D_LINUX})
    endif()
    # Assembler support was rewritten in CMake 2.8.5.
    # The assembler file must be passed to gcc but prior to this
    # version it is passed to as. This results in a bunch of
    # error message. This is only critical for non-x86 platforms.
    # On x86/x86-64 the file can safely be ignored.
    if("${CMAKE_VERSION}" MATCHES "^2\\.8\\.[01234]($|\\..*)")
        message(WARNING "Excluding core/threadasm.S from build because of missing CMake support.")
        message(WARNING "This file is required for certain non-x86 platforms.")
        message(WARNING "Please consider updating CMake to at least 2.8.5.")
    else()
        list(APPEND DCRT_ASM ${RUNTIME_DIR}/src/core/threadasm.S)
    endif()
    if(APPLE)
        list(APPEND CORE_D_SYS ${CORE_D_OSX})
    endif()
    list(REMOVE_ITEM LDC_D ${RUNTIME_DIR}/src/ldc/eh2.d)
    list(REMOVE_ITEM DCRT_C ${RUNTIME_DIR}/src/rt/msvc.c)

    # Using CMAKE_SYSTEM_PROCESSOR might be inacurrate when somebody is
    # cross-compiling by just setting the tool executbles to a cross toolchain,
    # so just always include the file.
    list(APPEND DCRT_C ${RUNTIME_DIR}/src/ldc/arm_unwind.c)
elseif(WIN32)
    list(APPEND CORE_D_SYS ${CORE_D_WIN})
    if (MSVC)
        list(REMOVE_ITEM LDC_D ${RUNTIME_DIR}/src/ldc/eh.d)
    else()
        list(REMOVE_ITEM LDC_D ${RUNTIME_DIR}/src/ldc/eh2.d)
    endif()
    list(REMOVE_ITEM DCRT_C ${RUNTIME_DIR}/src/rt/monitor.c)
endif()
list(APPEND CORE_D ${CORE_D_INTERNAL} ${CORE_D_SYNC} ${CORE_D_SYS} ${CORE_D_STDC})
list(APPEND CORE_D ${LDC_D} ${RUNTIME_DIR}/src/object_.d)
file(GLOB CORE_C ${RUNTIME_DIR}/src/core/stdc/*.c)

if(PHOBOS2_DIR)
    if(BUILD_SHARED_LIBS)
        # std.net.curl depends on libcurl – when building a shared library, we
        # need to take care of that.
        find_package(CURL REQUIRED)
    endif()

    file(GLOB PHOBOS2_D ${PHOBOS2_DIR}/std/*.d)
    file(GLOB PHOBOS2_D_CONTAINER ${PHOBOS2_DIR}/std/container/*.d)
    file(GLOB PHOBOS2_D_DIGEST ${PHOBOS2_DIR}/std/digest/*.d)
    file(GLOB PHOBOS2_D_EXPERIMENTAL ${PHOBOS2_DIR}/std/experimental/*.d)
    file(GLOB PHOBOS2_D_NET ${PHOBOS2_DIR}/std/net/*.d)
    file(GLOB_RECURSE PHOBOS2_D_INTERNAL ${PHOBOS2_DIR}/std/internal/*.d)
    file(GLOB PHOBOS2_D_C ${PHOBOS2_DIR}/std/c/*.d)
    file(GLOB PHOBOS2_ETC ${PHOBOS2_DIR}/etc/c/*.d)
    if(APPLE)
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/osx/*.d)
    elseif(UNIX)
        # Install Linux headers on all non-Apple *nixes - not correct, but
        # shouldn't cause any harm either.
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/linux/*.d)
    elseif(WIN32)
        file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/windows/*.d)
    endif()
    file(GLOB ZLIB_C ${PHOBOS2_DIR}/etc/c/zlib/*.c)
    list(REMOVE_ITEM ZLIB_C
        ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c
        ${PHOBOS2_DIR}/etc/c/zlib/example.c
        ${PHOBOS2_DIR}/etc/c/zlib/gzio.c
    )
    if(WIN32)
        file(GLOB PHOBOS2_D_WIN ${PHOBOS2_DIR}/std/windows/*.d)
    endif()
    list(APPEND PHOBOS2_D
            ${PHOBOS2_D_CONTAINER}
            ${PHOBOS2_D_DIGEST}
            ${PHOBOS2_D_EXPERIMENTAL}
            ${PHOBOS2_D_NET}
            ${PHOBOS2_D_INTERNAL}
            ${PHOBOS2_D_WIN}
            ${PHOBOS2_D_C}
            ${PHOBOS2_D_C_SYS}
            ${PHOBOS2_ETC}
    )
    list(REMOVE_ITEM PHOBOS2_D
            ${PHOBOS2_DIR}/std/intrinsic.d
    )
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
if(MULTILIB AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(MULTILIB_ADDITIONAL_PATH         "\n        \"-L-L${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",\n        \"-L-Xlinker -L--no-warn-search-mismatch\",")
    set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"-L-L${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",\n        \"-L-Xlinker -L--no-warn-search-mismatch\",")
endif()

if(BUILD_SHARED_LIBS AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(SHARED_LIBS_RPATH "\n        \"-L-rpath=${CMAKE_BINARY_DIR}/lib\",")
endif()

configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
# Prepare the config files we are going to install later in bin.
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)

#
# druntime/Phobos compilation helpers.
#

set(GCCBUILTINS "${PROJECT_BINARY_DIR}/gccbuiltins_x86.di")
add_custom_command(
    OUTPUT ${GCCBUILTINS}
    COMMAND gen_gccbuiltins ${GCCBUILTINS} "x86"
)

# Always build zlib and other C parts of the runtime in release mode, regardless
# of what the user chose for LDC itself. We can't easily do this on MSVC builds
# due to incompatible switches.
if(NOT MSVC)
    set(cfiles "${CORE_C};${DCRT_C}")
    if(PHOBOS2_DIR)
        list(APPEND cfiles "${ZLIB_C}")
    endif()
    set_source_files_properties(${cfiles} PROPERTIES
        COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}"
    )
endif()

# Compiles the given D module into an object file, and if enabled, a bitcode
# file. The ouput is written to a path based on output_dir. The paths of the
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

    # We hit a few known bugs in older LLVM versions when building the unit
    # tests with full optimizations. Downgrade the optimization flags for those
    # specific files so we can at least test the rest of them.
    set(cur_d_flags ${d_flags})
    if("${cur_d_flags}" MATCHES "-unittest")
        if(${LDC_LLVM_VER} EQUAL 301 AND "${input_d}" MATCHES "std/exception.d" AND
            ${HOST_BITNESS} EQUAL 64)
            # Building the std.exception tests on x86_64 triggers an infinite
            # recursion in scalar evolution on LLVM 3.1 (only), see the list of
            # known LLVM bugs for details.
            string(REPLACE "-O3" "-O1" cur_d_flags "${cur_d_flags}")
        elseif(${LDC_LLVM_VER} LESS 303 AND "${input_d}" MATCHES "std/range.d")
            # Building the std.range tests triggers an assertion error on
            # every LLVM release before 3.3. See PR15608.
            string(REPLACE "-O3" "-O1" cur_d_flags "${cur_d_flags}")
        elseif(${LDC_LLVM_VER} LESS 304 AND "${input_d}" MATCHES "rt/util/container/array.d")
            # Building the rt.util.container.array tests triggers a bug in the
            # jump threading pass on every LLVM release before 3.4.
            # See PR17621.
            string(REPLACE "-O3" "-O1" cur_d_flags "${cur_d_flags}")
        endif()
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
        COMMAND ${LDC_EXE} ${dc_flags} -c -I${RUNTIME_DIR}/src -I${RUNTIME_DIR}/src/gc ${input_d} -of${output_o} ${cur_d_flags}
        WORKING_DIRECTORY ${PROJECT_PARENT_DIR}
        DEPENDS ${input_d}
                ${LDC_EXE}
                ${GCCBUILTINS}
                ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf
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

    # Always disable invariants for debug builds of core.* and gc.* (there
    # are/were some broken invariants around; druntime is always built in
    # release mode in upstream builds).
    set(rt_flags "${d_flags};-disable-invariants")
    if(BUILD_SHARED_LIBS)
        set(shared ";-d-version=Shared")
    else()
        set(shared)
    endif()

    foreach(f ${CORE_D} ${GC_D})
        dc(
            ${f}
            "${rt_flags}${shared}"
            "${RUNTIME_DIR}"
            "${target_suffix}"
            ${outlist_o}
            ${outlist_bc}
        )
    endforeach()

    foreach(f ${DCRT_D})
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

# Builds a copy of druntime/Phobos from the source files gathered above. The
# names of the added library targets are appended to outlist_targets.
macro(build_runtime d_flags c_flags ld_flags lib_suffix path_suffix outlist_targets)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    set(druntime_o "")
    set(druntime_bc "")
    compile_druntime("${d_flags}" "${lib_suffix}" "${path_suffix}" druntime_o druntime_bc)

    add_library(druntime-ldc${target_suffix} ${D_LIBRARY_TYPE}
        ${druntime_o} ${CORE_C} ${DCRT_C} ${DCRT_ASM})
    set_target_properties(
        druntime-ldc${target_suffix} PROPERTIES
        OUTPUT_NAME                 druntime-ldc${lib_suffix}
        VERSION                     ${DMDFE_VERSION}
        SOVERSION                   ${DMDFE_PATCH_VERSION}
        LINKER_LANGUAGE             C
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${ld_flags}"
    )

    # When building a shared library, we need to link in all the default
    # libraries otherwise implicitly added by LDC to make it loadable from
    # C executables.
    if(BUILD_SHARED_LIBS)
        # TOOD: Once BUILD_SHARED_LIBS is supported on operating systems other
        # than Linux, the library selection will need to be adapted based on
        # the target.
        target_link_libraries(druntime-ldc${target_suffix} "m" "pthread" "rt" "dl")
    endif()

    list(APPEND ${outlist_targets} druntime-ldc${target_suffix})

    if(PHOBOS2_DIR)
        set(phobos2_o "")
        set(phobos2_bc "")
        compile_phobos2("${d_flags}" "${lib_suffix}" "${path_suffix}" phobos2_o phobos2_bc)

        add_library(phobos2-ldc${target_suffix} ${D_LIBRARY_TYPE} ${ZLIB_C} ${phobos2_o})
        set_target_properties(
            phobos2-ldc${target_suffix} PROPERTIES
            OUTPUT_NAME                 phobos2-ldc${lib_suffix}
            VERSION                     ${DMDFE_VERSION}
            SOVERSION                   ${DMDFE_PATCH_VERSION}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags}"
            LINK_FLAGS                  "${ld_flags}"
        )

        if(BUILD_SHARED_LIBS)
            # TODO: As for druntime, adapt once shared libraries are supported
            # on more operating systems.
            target_link_libraries(phobos2-ldc${target_suffix}
                druntime-ldc${target_suffix} "curl" "m" "pthread" "dl")
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
endmacro()

#
# Set up build targets.
#

if(MSVC)
    if (LINK_WITH_MSVCRT)
        set(RT_CFLAGS "/MDd")
    else()
        set(RT_CFLAGS "/MTd")
    endif()
else()
    set(RT_CFLAGS "")
endif()


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

        set(multitargets)
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" multitargets)

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
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" dummy)
    endif()
else()
    build_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
endif()

# Copy over druntime modules, preferring hand-written .di files.
set(DRUNTIME_IMPORT_DIR ${CMAKE_BINARY_DIR}/import)
set(DRUNTIME_PACKAGES core etc ldc)

set(druntime_modules)
foreach(p ${DRUNTIME_PACKAGES})
    file(GLOB_RECURSE m ${RUNTIME_DIR}/src/${p}/*.d ${RUNTIME_DIR}/src/${p}/*.di)
    list(APPEND druntime_modules ${m})
endforeach()

list(APPEND druntime_modules ${RUNTIME_DIR}/src/object.di)
foreach(f ${druntime_modules})
    if (NOT EXISTS "${f}i")
        file(RELATIVE_PATH relpath ${RUNTIME_DIR}/src ${f})
        configure_file(${f} ${DRUNTIME_IMPORT_DIR}/${relpath} COPYONLY)
    endif()
endforeach()


#
# Install target.
#

install(FILES ${DRUNTIME_IMPORT_DIR}/object.di DESTINATION ${INCLUDE_INSTALL_DIR}/ldc)
foreach(p ${DRUNTIME_PACKAGES})
    install(DIRECTORY ${DRUNTIME_IMPORT_DIR}/${p} DESTINATION ${INCLUDE_INSTALL_DIR})
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
# tests. They are invoked with the modules to test later. When using a shared
# runtime, we just build another copy of the two libraries with -unittest
# enabled. When linking statically, we have to directly pass the object files
# to the linking command instead so that all tests are pulled in.

macro(build_test_runner name_suffix d_flags c_flags)
    set(flags "${D_FLAGS};${d_flags};-unittest")
    if(BUILD_SHARED_LIBS)
        set(unittest_libs "")
        build_runtime(
            "${flags}"
            "${RT_CFLAGS} ${c_flags}"
            "${LD_FLAGS} ${c_flags}"
            "-unittest${name_suffix}"
            ""
            unittest_libs
        )

        # Only build the unittest libraries when running the tests. Unfortunately,
        # I couldn't find an easier way to make a test depend on a CMake target than
        # just invoking the build command through the CMake executable.
        set_target_properties(${unittest_libs} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)
        foreach(l ${unittest_libs})
            add_test(build-${l} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${l})
        endforeach()

        set(libarg "druntime-ldc-unittest${name_suffix}")
        add_test(NAME build-druntime-test-runner${name_suffix}
            COMMAND ${LDC_EXE}
                -of${PROJECT_BINARY_DIR}/druntime-test-runner${name_suffix}${CMAKE_EXECUTABLE_SUFFIX}
                -defaultlib=${libarg} -debuglib=${libarg}
                -singleobj ${flags} ${RUNTIME_DIR}/src/test_runner.d
        )
        set_tests_properties(build-druntime-test-runner${name_suffix} PROPERTIES
            DEPENDS build-druntime-ldc-unittest${name_suffix})

        if(PHOBOS2_DIR)
            set(libarg "phobos2-ldc-unittest${name_suffix},druntime-ldc-unittest${name_suffix}")
            add_test(NAME build-phobos2-test-runner${name_suffix}
                COMMAND ${LDC_EXE}
                    -of${PROJECT_BINARY_DIR}/phobos2-test-runner${name_suffix}${CMAKE_EXECUTABLE_SUFFIX}
                    -L--no-as-needed -defaultlib=${libarg} -debuglib=${libarg}
                    -singleobj ${flags} ${RUNTIME_DIR}/src/test_runner.d
            )
            set_tests_properties(build-phobos2-test-runner${name_suffix} PROPERTIES
                DEPENDS build-phobos2-ldc-unittest${name_suffix})
        endif()
    else()
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${flags}" "-unittest${name_suffix}" "" druntime_o druntime_bc)

        # We need to compile a small static library with the C/ASM files, as
        # there is no easy way of getting the object file names used when
        # compiling the main libraries. If we could require CMake 2.8.8, we
        # would be able to just build the files once (resp. twice for multilib)
        # for both the runtime debug/release builds and the tests. Oh well.
        set(druntime-casm druntime-ldc-casm${name_suffix})
        add_library(${druntime-casm} STATIC
            ${CORE_C} ${DCRT_C} ${DCRT_ASM})
        set_target_properties(
            ${druntime-casm} PROPERTIES
            LINKER_LANGUAGE             C
            COMPILE_FLAGS               "${RT_CFLAGS} ${c_flags}"
            LINK_FLAGS                  "${LD_FLAGS} ${ld_flags}"
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        )

        # See shared library case for explanation.
        set_target_properties(${druntime-casm} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)
        add_custom_target(druntime-ldc-unittest${name_suffix} DEPENDS ${druntime_o} ${druntime-casm})
        add_test(build-druntime-ldc-unittest${name_suffix} "${CMAKE_COMMAND}"
            --build ${CMAKE_BINARY_DIR} --target druntime-ldc-unittest${name_suffix})

        add_test(NAME build-druntime-test-runner${name_suffix}
            COMMAND ${LDC_EXE}
                -of${PROJECT_BINARY_DIR}/druntime-test-runner${name_suffix}${CMAKE_EXECUTABLE_SUFFIX}
                -defaultlib=${druntime-casm} -debuglib=${druntime-casm}
                -singleobj ${flags} ${druntime_o} ${RUNTIME_DIR}/src/test_runner.d
        )
        set_tests_properties(build-druntime-test-runner${name_suffix} PROPERTIES
            DEPENDS build-druntime-ldc-unittest${name_suffix}
        )

        # And the same for Phobos.
        if(PHOBOS2_DIR)
            set(phobos2_o "")
            set(phobos2_bc "")
            compile_phobos2("${flags}" "-unittest${name_suffix}" "" phobos2_o phobos2_bc)

            set(phobos2-casm phobos2-ldc-casm${name_suffix})
            add_library(${phobos2-casm} STATIC ${ZLIB_C})
            set_target_properties(
                ${phobos2-casm} PROPERTIES
                LINKER_LANGUAGE             C
                COMPILE_FLAGS               "${RT_CFLAGS} ${c_flags}"
                LINK_FLAGS                  "${LD_FLAGS} ${ld_flags}"
                ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            )

            set_target_properties(${phobos2-casm} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)
            add_custom_target(phobos2-ldc-unittest${name_suffix} DEPENDS ${phobos2_o} ${phobos2-casm})
            add_test(build-phobos2-ldc-unittest${name_suffix} "${CMAKE_COMMAND}"
                --build ${CMAKE_BINARY_DIR} --target phobos2-ldc-unittest${name_suffix})

            add_test(NAME build-phobos2-test-runner${name_suffix}
                COMMAND ${LDC_EXE}
                    -of${PROJECT_BINARY_DIR}/phobos2-test-runner${name_suffix}${CMAKE_EXECUTABLE_SUFFIX}
                    -defaultlib=druntime-ldc,${phobos2-casm} -debuglib=druntime-ldc,${phobos2-casm}
                    -singleobj -L-lcurl ${flags} ${phobos2_o} ${RUNTIME_DIR}/src/test_runner.d
            )
            set_tests_properties(build-phobos2-test-runner${name_suffix} PROPERTIES
                DEPENDS build-phobos2-ldc-unittest${name_suffix}
            )
        endif()
    endif()
endmacro()
build_test_runner("" "${D_FLAGS_RELEASE}" "")
build_test_runner("-debug" "${D_FLAGS_DEBUG}" "")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    build_test_runner("-32" "${D_FLAGS_RELEASE};-m32" "-m32")
    build_test_runner("-debug-32" "${D_FLAGS_DEBUG};-m32" "-m32")
endif()

# Add the druntime/Phobos test runner invocations for all the different modules.

macro(file_to_module_name file_name out_module_name)
    string(REPLACE ${PROJECT_SOURCE_DIR}/ "" stripped ${file_name})
    string(REPLACE "druntime/src/" "" stripped ${stripped})
    string(REPLACE "phobos/" "" stripped ${stripped})
    string(REPLACE "object_.d" "object.d" stripped ${stripped})
    string(REPLACE ".d" "" stripped ${stripped})
    string(REPLACE "/" "." module ${stripped})

    # The logical module name for package.d files doesn't include the
    # trailing .package part.
    string(REPLACE ".package" "" module ${module})

    # rt.invariant doesn't have a module declaration, presumably because
    # invariant is a keyword.
    string(REPLACE "rt.invariant" "invariant" ${out_module_name} ${module})
endmacro()

function(add_tests d_files runner name_suffix)
    foreach(file ${d_files})
        file_to_module_name(${file} module)
        if("${module}" STREQUAL "core.internal.convert")
            # Exclude unit tests for now, as they fail due to a CTFE vs. runtime
            # issue with floating point numbers. See the discussion on GitHub
            # pull request #770. To be revisited after 0.15.0 is out.
        else()
            add_test(NAME "${module}${name_suffix}"
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
                COMMAND ${runner}-test-runner${name_suffix} ${module}
            )
            set_tests_properties("${module}${name_suffix}" PROPERTIES
                DEPENDS build-${runner}-test-runner${name_suffix}
            )
        endif()
    endforeach()
endfunction()
function(add_runtime_tests name_suffix)
    add_tests("${CORE_D};${DCRT_D};${GC_D}" "druntime" "${name_suffix}")
    if(PHOBOS2_DIR)
        add_tests("${PHOBOS2_D}" "phobos2" "${name_suffix}")
    endif()
endfunction()

add_runtime_tests("")
add_runtime_tests("-debug")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    add_runtime_tests("-32")
    add_runtime_tests("-debug-32")
endif()

# Add the standalone druntime tests.
# TODO: Add test/excetions and test/init_fini.
if(BUILD_SHARED_LIBS)
    get_property(ldmd_path TARGET ldmd2 PROPERTY LOCATION)
    get_property(druntime_path TARGET druntime-ldc PROPERTY LOCATION)
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-shared)

    add_test(NAME clean-druntime-test-shared
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})
    add_test(NAME druntime-test-shared
        COMMAND make -C ${PROJECT_SOURCE_DIR}/druntime/test/shared
            ROOT=${outdir} DMD=${ldmd_path} MODEL=default DRUNTIMESO=${druntime_path}
            CFLAGS=-Wall\ -Wl,-rpath,${CMAKE_BINARY_DIR}/lib
    )
    set_tests_properties(druntime-test-shared PROPERTIES DEPENDS clean-druntime-test-shared)
endif()
