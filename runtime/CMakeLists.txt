project(runtime)

cmake_minimum_required(VERSION 2.6)

#
# Main configuration.
#

set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                       CACHE BOOL    "Build both 32/64 bit runtime libraries")
set(BUILD_BC_LIBS         OFF                                       CACHE BOOL    "Build the runtime as LLVM bitcode libraries")
set(LIB_SUFFIX            ""                                        CACHE STRING  "'64' to install libraries into ${PREFIX}/lib64")
set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d         CACHE PATH    "Path to install D modules to")
set(BUILD_SHARED_LIBS     OFF                                       CACHE BOOL    "Whether to build the runtime as a shared library (*UNSUPPORTED*)")
set(D_FLAGS               -w;-d                                     CACHE STRING  "Runtime build flags, separated by ;")
set(D_FLAGS_DEBUG         -g                                        CACHE STRING  "Runtime build flags (debug libraries), separated by ;")
set(D_FLAGS_RELEASE       -O3;-release                              CACHE STRING  "Runtime build flags (release libraries), separated by ;")
if(MSVC)
    set(LINK_WITH_MSVCRT  OFF                                       CACHE BOOL    "Link with MSVCRT.LIB instead of LIBCMT.LIB")
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

file(GLOB CORE_D ${RUNTIME_DIR}/src/core/*.d )
file(GLOB CORE_D_SYNC ${RUNTIME_DIR}/src/core/sync/*.d )
file(GLOB CORE_D_STDC ${RUNTIME_DIR}/src/core/stdc/*.d )
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
list(APPEND CORE_D ${CORE_D_SYNC} ${CORE_D_SYS} ${CORE_D_STDC})
list(APPEND CORE_D ${LDC_D} ${RUNTIME_DIR}/src/object_.d)
file(GLOB CORE_C ${RUNTIME_DIR}/src/core/stdc/*.c)

if(PHOBOS2_DIR)
    if(BUILD_SHARED_LIBS)
        # std.net.curl depends on libcurl – when building a shared library, we
        # need to take care of that.
        find_package(CURL REQUIRED)
    endif()

    file(GLOB PHOBOS2_D ${PHOBOS2_DIR}/std/*.d)
    file(GLOB PHOBOS2_D_DIGEST ${PHOBOS2_DIR}/std/digest/*.d)
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
            ${PHOBOS2_D_DIGEST}
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
if(NOT LDC_LOC)
    if(NOT LDC_EXE)
        set(LDC_EXE ldc2)
    endif()

    find_program(LDC_LOC ${LDC_EXE} ${PROJECT_BINARY_DIR}/../bin DOC "path to ldc binary")
    if(NOT LDC_LOC)
        message(SEND_ERROR "ldc not found")
    endif()
    set(LDC_EXE_NAME ${LDC_EXE})
endif()

#
# Create configuration files.
#

# Add extra paths on Linux and disable linker arch mismatch warnings (like
# DMD and GDC do). OS X doesn't need extra configuration due to the use of
# fat binaries. Other Posixen might need to be added here.
if(MULTILIB AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(MULTILIB_ADDITIONAL_PATH         "\n        \"-L-L${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
    set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"-L-L${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
endif()

configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
# Prepare the config files for installation in bin.
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.rebuild.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.rebuild.conf)

#
# Macros.
#

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

    set(output_o ${PROJECT_BINARY_DIR}/${output_root}${output_suffix}${CMAKE_C_OUTPUT_EXTENSION})
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
        COMMAND ${LDC_LOC} ${dc_flags} -c -I${RUNTIME_DIR}/src -I${RUNTIME_DIR}/src/gc ${input_d} -of${output_o} ${d_flags}
        WORKING_DIRECTORY ${PROJECT_PARENT_DIR}
        DEPENDS ${LDC_LOC}
            ${input_d}
            ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf
    )
endmacro()

# Builds a copy of druntime/Phobos from the source files gathered above. The
# names of the added library targets are appended to outlist_targets.
macro(build_runtime d_flags c_flags ld_flags lib_suffix path_suffix outlist_targets)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    # "Vanity" suffix for target names.
    set(target_suffix "")
    if(NOT "${lib_suffix}" STREQUAL "")
        set(target_suffix "${lib_suffix}")
    endif()
    if(NOT "${path_suffix}" STREQUAL "")
        set(target_suffix "${target_suffix}_${path_suffix}")
    endif()

    # Always disable invariants for debug builds of core.* and gc.* (there
    # are/were some broken invariants around; druntime is always built in
    # release mode in upstream builds).
    set(rt_flags "${d_flags};-disable-invariants")

    set(CORE_O "")
    set(CORE_BC "")
    foreach(f ${CORE_D})
        dc(${f} "${rt_flags}" "${RUNTIME_DIR}" "${target_suffix}" CORE_O CORE_BC)
    endforeach()

    set(GC_O "")
    set(GC_BC "")
    foreach(f ${GC_D})
        dc(${f} "${rt_flags}" "${RUNTIME_DIR}" "${target_suffix}" GC_O GC_BC)
    endforeach()

    set(DCRT_O "")
    set(DCRT_BC "")
    foreach(f ${DCRT_D})
        dc(${f} "${d_flags}" "${RUNTIME_DIR}" "${target_suffix}" DCRT_O DCRT_BC)
    endforeach()

    if(NOT MSVC)
        # Always build zlib and other C parts of the runtime in release mode.
        set_source_files_properties(${CORE_C} ${DCRT_C} PROPERTIES
            COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}"
        )
    endif()

    set(GCCBUILTINS "${PROJECT_BINARY_DIR}/gccbuiltins_x86.di")
    add_custom_command(
        OUTPUT ${GCCBUILTINS}
        COMMAND ${GEN_GCCBUILTINS_LOC} ${GCCBUILTINS} "x86"
        DEPENDS ${GEN_GCCBUILTINS_LOC}
    )
    add_library(druntime-ldc${target_suffix}
                ${D_LIBRARY_TYPE}
                ${CORE_O}
                ${CORE_C}
                ${GC_O}
                ${DCRT_O}
                ${DCRT_C}
                ${DCRT_ASM}
                ${GCCBUILTINS}
    )
    set_target_properties(druntime-ldc${target_suffix} PROPERTIES OUTPUT_NAME druntime-ldc${lib_suffix})

    set_target_properties(
        druntime-ldc${target_suffix} PROPERTIES
        VERSION                     ${DMDFE_VERSION}
        SOVERSION                   ${DMDFE_PATCH_VERSION}
        LINKER_LANGUAGE             C
        ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
        LIBRARY_OUTPUT_DIRECTORY    ${output_path}
        RUNTIME_OUTPUT_DIRECTORY    ${output_path}
        COMPILE_FLAGS               "${c_flags}"
        LINK_FLAGS                  "${ld_flags}"
    )
    list(APPEND ${outlist_targets} druntime-ldc${target_suffix})

    if(PHOBOS2_DIR)
        set(PHOBOS2_O "")
        set(PHOBOS2_BC "")
        foreach(f ${PHOBOS2_D})
             dc(${f} "${d_flags};-I${PHOBOS2_DIR}" ${PHOBOS2_DIR} "${target_suffix}" PHOBOS2_O PHOBOS2_BC)
        endforeach()

        add_library(phobos2-ldc${target_suffix} ${D_LIBRARY_TYPE}
            ${ZLIB_C}
            ${PHOBOS2_O}
        )
        set_target_properties(
            phobos2-ldc${target_suffix} PROPERTIES
            VERSION                     ${DMDFE_VERSION}
            SOVERSION                   ${DMDFE_PATCH_VERSION}
            OUTPUT_NAME                 phobos2-ldc${lib_suffix}
            LINKER_LANGUAGE             C
            ARCHIVE_OUTPUT_DIRECTORY    ${output_path}
            LIBRARY_OUTPUT_DIRECTORY    ${output_path}
            RUNTIME_OUTPUT_DIRECTORY    ${output_path}
            COMPILE_FLAGS               "${c_flags}"
            LINK_FLAGS                  "${ld_flags}"
        )

        if(BUILD_SHARED_LIBS)
            target_link_libraries(phobos2-ldc${target_suffix} druntime-ldc${target_suffix})
            target_link_libraries(phobos2-ldc${target_suffix} "curl")
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
            COMMAND ${LLVM_AR_EXE} rs libdruntime-ldc${lib_suffix}-bc.a ${CORE_BC} ${GC_BC} ${DCRT_BC}
            COMMAND ${LLVM_AR_EXE} rs libphobos2-ldc${lib_suffix}-bc.a ${PHOBOS2_BC}
            WORKING_DIRECTORY ${output_path}
            DEPENDS
                ${CORE_BC}
                ${GC_BC}
                ${DCRT_BC}
                ${PHOBOS2_BC}
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
        build_runtime_variants("" "${RT_CLAGS}" "${LD_FLAGS}" "${hostsuffix}" hosttargets)

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
        build_runtime_variants("" "${RT_CLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
        build_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" dummy)
    endif()
else()
    build_runtime_variants("" "${RT_CLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" LIBS_TO_INSTALL)
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
function(add_tests module_files)
    foreach(file ${module_files})
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" stripped ${file})
        string(REPLACE ".d" "" stripped ${stripped})
        string(REPLACE "/" "_" testroot ${stripped})

        function(testcase name flags)
            # -singleobj to avoid output file clashes when tests are run in parallel.
            add_test(NAME ${testroot}_${name}_build
                COMMAND ${LDC_LOC}
                    -of${PROJECT_BINARY_DIR}/${testroot}_${name}
                    -unittest -main -d -w -singleobj ${flags} ${file}
            )
            add_test(NAME ${testroot}_${name}_run COMMAND ${PROJECT_BINARY_DIR}/${testroot}_${name})
            set_tests_properties(${testroot}_${name}_run PROPERTIES DEPENDS ${testroot}_${name}_build)
        endfunction()

        set(dbgflags "-g;-d-debug")
        if("${testroot}" STREQUAL "phobos_std_net_curl")
            set(dbgflags "${dbgflags};-L=-lcurl")
        endif()
        testcase(debug "${dbgflags}")

        set(relflags "-O3;-release")
        # Building the std.exception tests on x86_64 triggers an infinite
        # recursion in scalar evolution on LLVM 3.1 (only), see the list of
        # known LLVM bugs for details.
        if(${LDC_LLVM_VER} EQUAL 301 AND ${HOST_BITNESS} EQUAL 64 AND
            "${testroot}" STREQUAL "phobos_std_exception")
            set(relflags "-O1;-release")
        # Building the std.range tests triggers an assertion error on
        # every LLVM release before 3.3. See PR15608.
        elseif(${LDC_LLVM_VER} LESS 303 AND
            "${testroot}" STREQUAL "phobos_std_range")
            set(relflags "-O1;-release")
        # Building the rt.util.container tests triggers a bug in the
        # jump threading pass on every LLVM release before 3.4.
        # See PR17621.
        elseif(${LDC_LLVM_VER} LESS 304 AND
            "${testroot}" STREQUAL "druntime_src_rt_util_container")
            set(relflags "-O1;-release")
        endif()
        if("${testroot}" STREQUAL "phobos_std_net_curl")
            set(relflags "${relflags};-L=-lcurl")
        endif()
        testcase(release "${relflags}")

        # On 64 bit multilib builds, run the tests in 32 bit mode as well.
        if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
            testcase(debug_32 "${dbgflags};-m32")
            testcase(release_32 "${relflags};-m32")
        endif()
    endforeach()
endfunction()

add_tests("${CORE_D}")
add_tests("${DCRT_D}")
add_tests("${GC_D}")
if(PHOBOS2_DIR)
    add_tests("${PHOBOS2_D}")
endif()
