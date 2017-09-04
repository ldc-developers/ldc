project(runtime)

cmake_minimum_required(VERSION 2.8.9)

# Verify required variables if this CMake project is NOT embedded in the LDC CMake project.
if(NOT LDC_EXE)
    if(NOT LDC_EXE_FULL)
        message(FATAL_ERROR "Please define the path to the LDC executable via -DLDC_EXE_FULL=...")
    endif()
    if(NOT DEFINED D_VERSION OR NOT DEFINED DMDFE_MINOR_VERSION OR NOT DEFINED DMDFE_PATCH_VERSION)
        message(FATAL_ERROR "Please define the CMake variables D_VERSION, DMDFE_MINOR_VERSION and DMDFE_PATCH_VERSION.")
    endif()

    # Helper function
    function(append value)
        foreach(variable ${ARGN})
            if(${variable} STREQUAL "")
                set(${variable} "${value}" PARENT_SCOPE)
            else()
                set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
            endif()
        endforeach(variable)
    endfunction()
endif()

#
# Main configuration.
#

set(DMDFE_VERSION         ${D_VERSION}.${DMDFE_MINOR_VERSION}.${DMDFE_PATCH_VERSION})

set(MULTILIB              OFF                                 CACHE BOOL   "Build both 32/64 bit runtime libraries")
set(BUILD_BC_LIBS         OFF                                 CACHE BOOL   "Build the runtime as LLVM bitcode libraries")

if(NOT DEFINED INCLUDE_DIR_RELATIVE_TO_LDCBINARY_PATH)
	if((NOT DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
		set(INCLUDE_INSTALL_DIR		"/usr/include/d")
	elseif((DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
		set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR})
	elseif((NOT DEFINED INCLUDE_INSTALL_DIR) AND (DEFINED CMAKE_INSTALL_PREFIX))
		set(INCLUDE_INSTALL_DIR		${CMAKE_INSTALL_PREFIX}/include/d)
	else()
		set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR})
	endif()
	set(INCLUDE_INSTALL_DIR	  ${INCLUDE_INSTALL_DIR}	  	  CACHE PATH "Path to install D modules to" FORCE)
	set(LDC2_INCLUDE_DIR	  ${INCLUDE_INSTALL_DIR}		  CACHE PATH "Include path that will be set in ldc2.conf")
	message(STATUS "LDC2_INCLUDE_DIR: ${LDC2_INCLUDE_DIR}")
else()
	if(NOT DEFINED INCLUDE_DIR)
		if((NOT DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		"/usr/include/d")
		elseif((DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR})
		elseif((NOT DEFINED INCLUDE_INSTALL_DIR) AND (DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		${CMAKE_INSTALL_PREFIX}/include/d)
		else()
			set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR})
		endif()
	else()
		set(INCLUDE_DIR		  			${INCLUDE_DIR}			  CACHE STRING "Modifiable path relative to CMAKE_INSTALL_PREFIX or INCLUDE_INSTALL_DIR for include files")
		if((NOT DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		/usr/${INCLUDE_DIR})
		elseif((DEFINED INCLUDE_INSTALL_DIR) AND (NOT DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR}/${INCLUDE_DIR})
		elseif((NOT DEFINED INCLUDE_INSTALL_DIR) AND (DEFINED CMAKE_INSTALL_PREFIX))
			set(INCLUDE_INSTALL_DIR		${CMAKE_INSTALL_PREFIX}/${INCLUDE_DIR})
		else()
			set(INCLUDE_INSTALL_DIR		${INCLUDE_INSTALL_DIR}/${INCLUDE_DIR})
		endif()
	endif()
	set(INCLUDE_INSTALL_DIR				${INCLUDE_INSTALL_DIR}	  CACHE PATH "Path to install D modules to" FORCE)
	#Set LDC2_INCLUDE_DIR based upon INCLUDE_INSTALL_DIR and CMAKE_INSTALL_PREFIX#########################################
	if(NOT IS_ABSOLUTE ${INCLUDE_INSTALL_DIR})
		get_filename_component(INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR} REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})
	endif()

	string(REGEX REPLACE "^/" "" list_include_install_dir ${INCLUDE_INSTALL_DIR})
	string(REGEX REPLACE "^/" "" list_cmake_install_prefix ${CMAKE_INSTALL_PREFIX})
	string(REGEX REPLACE "/" ";" list_include_install_dir ${list_include_install_dir})
	string(REGEX REPLACE "/" ";" list_cmake_install_prefix ${list_cmake_install_prefix})

	list(LENGTH list_cmake_install_prefix list_length_cmake_install_prefix)
	list(LENGTH list_include_install_dir list_length_include_install_dir)

	set(list_index "0")
	foreach(subdir_a ${list_cmake_install_prefix})
		list(GET list_include_install_dir ${list_index} subdir_b)
		if(NOT subdir_a STREQUAL subdir_b)
			if(NOT DEFINED ${common_subdir_count})
				MATH(EXPR common_subdir_count "${common_subdir_count}+0")
			endif()
			MATH(EXPR uncommon_subdir_count "${list_length_cmake_install_prefix}-${common_subdir_count}-1")
			break()
		else()
			MATH(EXPR common_subdir_count "${common_subdir_count}+1")
		endif()
		MATH(EXPR list_index "${list_index}+1")
	endforeach()

	MATH(EXPR max_count "${list_length_include_install_dir}-1")
	set(LDC2_INCLUDE_DIR "%%ldcbinarypath%%/../")

	if(DEFINED uncommon_subdir_count)
		foreach(count RANGE ${uncommon_subdir_count})
			list(APPEND LDC2_INCLUDE_DIR "../")
		endforeach()
	endif()
	foreach(subdir_index RANGE ${common_subdir_count} ${max_count})
		list(GET list_include_install_dir ${subdir_index} subdir)
		list(APPEND LDC2_INCLUDE_DIR "${subdir}/")
	endforeach()
	string(REGEX REPLACE ";" "" LDC2_INCLUDE_DIR ${LDC2_INCLUDE_DIR})
	string(REGEX REPLACE "/$" "" LDC2_INCLUDE_DIR ${LDC2_INCLUDE_DIR})
	set(LDC2_INCLUDE_DIR	  ${LDC2_INCLUDE_DIR}			  CACHE PATH "Include path that will be set in ldc2.conf")

	message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
	message(STATUS "INCLUDE_INSTALL_DIR: ${INCLUDE_INSTALL_DIR}")
	message(STATUS "CMAKE_INSTALL_PREFIX subdir list: ${list_cmake_install_prefix}")
	message(STATUS "INCLUDE_INSTALL_DIR subdir list: ${list_include_install_dir}")
	message(STATUS "CMAKE SUBDIR COUNT: ${list_length_cmake_install_prefix}")
	message(STATUS "COMMON SUBDIR COUNT: ${common_subdir_count}")
	message(STATUS "LDC2_INCLUDE_DIR: ${LDC2_INCLUDE_DIR}")
endif()
######################################################################################################################

set(BUILD_SHARED_LIBS     AUTO                                CACHE STRING "Whether to build the runtime as a shared library (ON|OFF|BOTH)")
set(D_FLAGS               -w                                  CACHE STRING "Runtime D compiler flags, separated by ';'")
set(D_FLAGS_DEBUG         -g;-link-debuglib                   CACHE STRING "Runtime D compiler flags (debug libraries), separated by ';'")
set(D_FLAGS_RELEASE       -O3;-release                        CACHE STRING "Runtime D compiler flags (release libraries), separated by ';'")
set(COMPILE_ALL_D_FILES_AT_ONCE ON                            CACHE BOOL   "Compile all D files for a lib in a single command line instead of separately")
set(RT_CFLAGS             ""                                  CACHE STRING "Runtime extra C compiler flags, separated by ' '")
set(LD_FLAGS              ""                                  CACHE STRING "Runtime extra C linker flags, separated by ' '")
set(C_SYSTEM_LIBS         AUTO                                CACHE STRING "C system libraries for linking shared libraries and test runners, separated by ';'")
set(TARGET_SYSTEM         AUTO                                CACHE STRING "Targeted platform for cross-compilation (e.g., 'Linux;UNIX', 'Darwin;APPLE;UNIX', 'Windows;MSVC')")

set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

# Auto-detect TARGET_SYSTEM
if("${TARGET_SYSTEM}" STREQUAL "AUTO")
    set(TARGET_SYSTEM ${CMAKE_SYSTEM_NAME})
    # Additionally add MSVC, APPLE and/or UNIX
    if(MSVC)
        list(APPEND TARGET_SYSTEM "MSVC")
    endif()
    if(APPLE)
        list(APPEND TARGET_SYSTEM "APPLE")
    endif()
    if(UNIX)
        list(APPEND TARGET_SYSTEM "UNIX")
    endif()
endif()

if("${TARGET_SYSTEM}" MATCHES "UNIX")
    ENABLE_LANGUAGE(ASM)
elseif("${TARGET_SYSTEM}" MATCHES "MSVC")
    ENABLE_LANGUAGE(ASM_MASM)
endif()

include(CheckTypeSize)
check_type_size(void* ptr_size)
if(${ptr_size} MATCHES "^8$") ## if it's 64-bit OS
    set(HOST_BITNESS 64)
    set(MULTILIB_SUFFIX 32)
else()
    set(HOST_BITNESS 32)
    set(MULTILIB_SUFFIX 64)
endif()

set(SHARED_LIBS_SUPPORTED OFF)
if("${TARGET_SYSTEM}" MATCHES "Linux|FreeBSD|APPLE"
   AND (NOT "${TARGET_SYSTEM}" MATCHES "Android"))
    set(SHARED_LIBS_SUPPORTED ON)
endif()

if(${BUILD_SHARED_LIBS} STREQUAL "AUTO")
    if(SHARED_LIBS_SUPPORTED)
        set(BUILD_SHARED_LIBS BOTH)
    else()
        set(BUILD_SHARED_LIBS OFF)
    endif()
endif()

set(SHARED_LIB_SUFFIX "")
if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
    if(NOT SHARED_LIBS_SUPPORTED)
        message(FATAL_ERROR "Shared libraries (BUILD_SHARED_LIBS) are only supported on Linux, macOS and FreeBSD for the time being.")
    endif()

    # Only use the `-shared` lib suffix if static libs are generated too
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(SHARED_LIB_SUFFIX "-shared")
    endif()
endif()

# Auto-detect C system libraries
if("${C_SYSTEM_LIBS}" STREQUAL "AUTO")
    if("${TARGET_SYSTEM}" MATCHES "Android")
        set(C_SYSTEM_LIBS m c)
    elseif("${TARGET_SYSTEM}" MATCHES "Linux")
        set(C_SYSTEM_LIBS m pthread rt dl)
    else()
        set(C_SYSTEM_LIBS m pthread)
    endif()
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
file(GLOB_RECURSE DRUNTIME_D_BIONIC  ${RUNTIME_DIR}/src/core/sys/bionic/*.d)
file(GLOB_RECURSE DRUNTIME_D_DARWIN  ${RUNTIME_DIR}/src/core/sys/darwin/*.d)
file(GLOB_RECURSE DRUNTIME_D_FREEBSD ${RUNTIME_DIR}/src/core/sys/freebsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_LINUX   ${RUNTIME_DIR}/src/core/sys/linux/*.d)
file(GLOB_RECURSE DRUNTIME_D_NETBSD  ${RUNTIME_DIR}/src/core/sys/netbsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_OPENBSD ${RUNTIME_DIR}/src/core/sys/openbsd/*.d)
file(GLOB_RECURSE DRUNTIME_D_OSX     ${RUNTIME_DIR}/src/core/sys/osx/*.d)
file(GLOB_RECURSE DRUNTIME_D_POSIX   ${RUNTIME_DIR}/src/core/sys/posix/*.d)
file(GLOB_RECURSE DRUNTIME_D_SOLARIS ${RUNTIME_DIR}/src/core/sys/solaris/*.d)
file(GLOB_RECURSE DRUNTIME_D_WINDOWS ${RUNTIME_DIR}/src/core/sys/windows/*.d)
list(REMOVE_ITEM DRUNTIME_D
    ${DRUNTIME_D_BIONIC}  ${DRUNTIME_D_DARWIN}  ${DRUNTIME_D_FREEBSD} ${DRUNTIME_D_LINUX}
    ${DRUNTIME_D_NETBSD}  ${DRUNTIME_D_OPENBSD} ${DRUNTIME_D_OSX}     ${DRUNTIME_D_POSIX}
    ${DRUNTIME_D_SOLARIS} ${DRUNTIME_D_WINDOWS}
)
if("${TARGET_SYSTEM}" MATCHES "Windows")
    list(APPEND DRUNTIME_D ${DRUNTIME_D_WINDOWS})
elseif("${TARGET_SYSTEM}" MATCHES "UNIX")
    list(APPEND DRUNTIME_D ${DRUNTIME_D_POSIX})
    if("${TARGET_SYSTEM}" MATCHES "APPLE")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_DARWIN})
        list(APPEND DRUNTIME_D ${DRUNTIME_D_OSX})
    elseif("${TARGET_SYSTEM}" MATCHES "FreeBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_FREEBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "Linux")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_LINUX})
        list(APPEND DRUNTIME_D ${DRUNTIME_D_BIONIC})
    elseif("${TARGET_SYSTEM}" MATCHES "NetBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_NETBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "OpenBSD")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_OPENBSD})
    elseif("${TARGET_SYSTEM}" MATCHES "SunOS")
        list(APPEND DRUNTIME_D ${DRUNTIME_D_SOLARIS})
    endif()
endif()

# druntime C parts
file(GLOB_RECURSE DRUNTIME_C ${RUNTIME_DIR}/src/*.c)
list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/dylib_fixes.c)
if(NOT "${TARGET_SYSTEM}" MATCHES "APPLE")
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/osx_tls.c)
endif()
if(NOT "${TARGET_SYSTEM}" MATCHES "MSVC")
    list(REMOVE_ITEM DRUNTIME_C ${RUNTIME_DIR}/src/rt/msvc.c ${RUNTIME_DIR}/src/rt/msvc_math.c)
endif()

# druntime ASM parts
set(DRUNTIME_ASM)
if("${TARGET_SYSTEM}" MATCHES "UNIX")
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
    if("${TARGET_SYSTEM}" MATCHES "UNIX")
        if("${TARGET_SYSTEM}" MATCHES "APPLE")
            list(APPEND PHOBOS2_D ${PHOBOS2_D_OSX})
        elseif("${TARGET_SYSTEM}" MATCHES "FreeBSD")
            list(APPEND PHOBOS2_D ${PHOBOS2_D_FREEBSD})
        elseif("${TARGET_SYSTEM}" MATCHES "Linux")
            list(APPEND PHOBOS2_D ${PHOBOS2_D_LINUX})
        endif()
    elseif("${TARGET_SYSTEM}" MATCHES "Windows")
        list(APPEND PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()
    # only include std/windows/ modules on Windows
    if(NOT "${TARGET_SYSTEM}" MATCHES "Windows")
        file(GLOB_RECURSE PHOBOS2_D_WINDOWS ${PHOBOS2_DIR}/std/windows/*.d)
        list(REMOVE_ITEM PHOBOS2_D ${PHOBOS2_D_WINDOWS})
    endif()

    # Phobos C parts
    file(GLOB_RECURSE PHOBOS2_C ${PHOBOS2_DIR}/*.c)
    # remove some obsolete zlib modules
    list(REMOVE_ITEM PHOBOS2_C ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c ${PHOBOS2_DIR}/etc/c/zlib/example.c)
endif()

#
# Create configuration files.
#

# Add extra paths on Linux and disable linker arch mismatch warnings (like
# DMD and GDC do). OS X doesn't need extra configuration due to the use of
# fat binaries. Other Posixen might need to be added here.
if("${TARGET_SYSTEM}" MATCHES "Linux|FreeBSD")
    if(MULTILIB)
        set(MULTILIB_ADDITIONAL_PATH         "\n        \"-L-L${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
        set(MULTILIB_ADDITIONAL_INSTALL_PATH "\n        \"-L-L${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX}\",\n        \"-L--no-warn-search-mismatch\",")
    endif()

    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        if(MULTILIB)
            set(SHARED_LIBS_RPATH "\n        \"-L-rpath=${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}:${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}\",")
        else()
            set(SHARED_LIBS_RPATH "\n        \"-L-rpath=${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}\",")
        endif()
    endif()
endif()

# Only generate the config files if this CMake project is embedded in the LDC CMake project.
if(LDC_EXE)
    if(PHOBOS2_DIR)
        set(CONFIG_NAME ${LDC_EXE}_phobos)
    else()
        set(CONFIG_NAME ${LDC_EXE})
    endif()
    configure_file(${PROJECT_PARENT_DIR}/${CONFIG_NAME}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf @ONLY)
    configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf @ONLY)
    configure_file(${PROJECT_SOURCE_DIR}/ldc-build-runtime.d.in ${PROJECT_BINARY_DIR}/../ldc-build-runtime.d @ONLY)
endif()

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
if("${TARGET_SYSTEM}" MATCHES "MSVC")
    # Omit all references to the default C runtime libs.
    # We want to be able to link against all 4 C runtime variants (libcmt[d] / msvcrt[d]).
    string(REGEX REPLACE "(^| )[/-]M[TD]d?( |$)" "\\2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    append("/MT /Zl" CMAKE_C_FLAGS_RELEASE)
    # warning C4131: uses old-style declarator
    # warning C4206: nonstandard extension used: translation unit is empty
    # warning C4996: zlib uses 'deprecated' POSIX names
    append("/wd4131 /wd4206 /wd4996" CMAKE_C_FLAGS_RELEASE)
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

# Compiles the given D modules to object files, and if enabled, bitcode files.
# The paths of the output files are appended to outlist_o and outlist_bc, respectively.
macro(dc src_files src_basedir d_flags output_basedir all_at_once outlist_o outlist_bc)
    set(dc_flags -c --output-o ${d_flags})
    if(BUILD_BC_LIBS)
        list(APPEND dc_flags --output-bc)
    endif()

    set(dc_deps ${LDC_EXE}
                ${LDC_EXE_FULL}
                ${GCCBUILTINS}
    )

    set(relative_src_files "")

    foreach(f ${src_files})
        file(RELATIVE_PATH relative_src_file ${src_basedir} ${f})
        get_filename_component(name ${relative_src_file} NAME_WE)
        get_filename_component(path ${relative_src_file} PATH)
        if("${path}" STREQUAL "")
            set(output_root ${name})
        else()
            set(output_root ${path}/${name})
        endif()

        set(output_o  ${output_basedir}/${output_root}${CMAKE_C_OUTPUT_EXTENSION})
        set(output_bc ${output_basedir}/${output_root}.bc)
        list(APPEND ${outlist_o} ${output_o})
        if(BUILD_BC_LIBS)
            list(APPEND ${outlist_bc} ${output_bc})
        endif()

        if(${all_at_once})
            list(APPEND relative_src_files ${relative_src_file})
        else()
            set(outfiles ${output_o})
            if(BUILD_BC_LIBS)
                list(APPEND outfiles ${output_bc})
            endif()

            add_custom_command(
                OUTPUT  ${outfiles}
                COMMAND ${LDC_EXE_FULL} ${dc_flags} -of=${output_o} ${relative_src_file}
                WORKING_DIRECTORY ${src_basedir}
                DEPENDS ${f} ${dc_deps}
            )
        endif()
    endforeach()

    if(${all_at_once})
        add_custom_command(
            OUTPUT  ${${outlist_o}} ${${outlist_bc}}
            COMMAND ${LDC_EXE_FULL} ${dc_flags} -od=${output_basedir} -op ${relative_src_files}
            WORKING_DIRECTORY ${src_basedir}
            DEPENDS ${src_files} ${dc_deps}
        )
    endif()
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

# Sets up the target(s) for building the druntime D object files, appending the
# names of the (bitcode) files to link into the library to outlist_o (outlist_bc).
macro(compile_druntime d_flags lib_suffix path_suffix all_at_once outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    dc("${DRUNTIME_D}"
       "${RUNTIME_DIR}/src"
       "-conf=;${d_flags};-I${RUNTIME_DIR}/src"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${all_at_once}"
       ${outlist_o}
       ${outlist_bc}
    )
endmacro()

# Sets up the targets for building the Phobos D object files, appending the
# names of the (bitcode) files to link into the library to outlist_o (outlist_bc).
macro(compile_phobos2 d_flags lib_suffix path_suffix all_at_once outlist_o outlist_bc)
    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    dc("${PHOBOS2_D}"
       "${PHOBOS2_DIR}"
       "-conf=;${d_flags};-I${RUNTIME_DIR}/src;-I${PHOBOS2_DIR}"
       "${PROJECT_BINARY_DIR}/objects${target_suffix}"
       "${all_at_once}"
       ${outlist_o}
       ${outlist_bc}
    )
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
    if("${TARGET_SYSTEM}" MATCHES "Linux")
        set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

# Builds a copy of druntime/Phobos from the specified object (bitcode) files.
# The names of the added library targets are appended to outlist_targets.
macro(build_runtime_libs druntime_o druntime_bc phobos2_o phobos2_bc c_flags ld_flags
                         lib_suffix path_suffix is_shared outlist_targets)
    set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

    set(library_type STATIC)
    if("${is_shared}" STREQUAL "ON")
        set(library_type SHARED)
    endif()

    get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
    add_library(druntime-ldc${target_suffix} ${library_type}
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
    if("${is_shared}" STREQUAL "ON")
        target_link_libraries(druntime-ldc${target_suffix} ${C_SYSTEM_LIBS})
    endif()

    list(APPEND ${outlist_targets} druntime-ldc${target_suffix})

    if(PHOBOS2_DIR)
        add_library(phobos2-ldc${target_suffix} ${library_type}
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

        if("${is_shared}" STREQUAL "ON")
            target_link_libraries(phobos2-ldc${target_suffix}
                druntime-ldc${target_suffix} ${C_SYSTEM_LIBS})
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

# Builds a static and/or shared copy of druntime/Phobos.
macro(build_runtime_variant d_flags c_flags ld_flags lib_suffix path_suffix all_d_files_at_once outlist_targets)
    # Only compile Phobos modules once for static+shared libs.
    # If a shared Phobos lib is generated (too), compile explicitly with `-relocation-model=pic`.
    set(phobos2_d_flags "${d_flags}")
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        list(APPEND ${phobos2_d_flags} -relocation-model=pic)
    endif()

    set(phobos2_o "")
    set(phobos2_bc "")
    compile_phobos2("${phobos2_d_flags}" "${lib_suffix}" "${path_suffix}" "${all_d_files_at_once}" phobos2_o phobos2_bc)

    # Use a dummy custom target ('phobos2-ldc…-common') depending solely on the Phobos D objects
    # (custom commands) as dependency for static+shared Phobos libs.
    # Otherwise building both libs in parallel may result in conflicting Phobos module compilations
    # (at least with the make generator), a known CMake issue.
    set(phobos2_common phobos2-ldc${target_suffix}-common)
    add_custom_target(${phobos2_common} DEPENDS ${phobos2_o} ${phobos2_bc})

    # static druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${d_flags}" "${lib_suffix}" "${path_suffix}" "${all_d_files_at_once}" druntime_o druntime_bc)

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}" "${path_suffix}" "OFF"
                           ${outlist_targets})
        add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
    endif()
    # shared druntime/Phobos
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        set(druntime_o "")
        set(druntime_bc "")
        compile_druntime("${d_flags};-relocation-model=pic;-d-version=Shared" "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}" "${all_d_files_at_once}" druntime_o druntime_bc)

        build_runtime_libs("${druntime_o}" "${druntime_bc}" "${phobos2_o}" "${phobos2_bc}"
                           "${c_flags}" "${ld_flags}" "${lib_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}" "ON"
                           ${outlist_targets})
        add_dependencies(phobos2-ldc${target_suffix} ${phobos2_common})
    endif()
endmacro()

# Builds static and/or shared pairs of debug+release copies of druntime/Phobos.
macro(build_runtime_variants d_flags c_flags ld_flags path_suffix outlist_targets)
    # static and/or shared release druntime/Phobos
    build_runtime_variant(
        "${d_flags};${D_FLAGS};${D_FLAGS_RELEASE}"
        "${c_flags}"
        "${ld_flags}"
        ""
        "${path_suffix}"
        "${COMPILE_ALL_D_FILES_AT_ONCE}"
        ${outlist_targets}
    )
    # static and/or shared debug druntime/Phobos
    build_runtime_variant(
        "${d_flags};${D_FLAGS};${D_FLAGS_DEBUG}"
        "${c_flags}"
        "${ld_flags}"
        "-debug"
        "${path_suffix}"
        "${COMPILE_ALL_D_FILES_AT_ONCE}"
        ${outlist_targets}
    )
endmacro()

# Builds static and/or shared debug+release druntime/Phobos + static profile-rt.
macro(build_all_runtime_variants d_flags c_flags ld_flags path_suffix outlist_targets)
    build_runtime_variants("${d_flags}" "${c_flags}" "${ld_flags}" "${path_suffix}" ${outlist_targets})

    if(LDC_WITH_PGO)
        # static profile-rt
        build_profile_runtime("${d_flags};${D_FLAGS};${D_FLAGS_RELEASE}" "${c_flags}" "${ld_flags}" "" "${path_suffix}" ${outlist_targets})
        get_target_suffix("" "${path_suffix}" target_suffix)
        set_common_library_properties(ldc-profile-rt${target_suffix})
    endif()
endmacro()

# Setup the build of profile-rt
include(profile-rt/DefineBuildProfileRT.cmake)

#
# Set up build and install targets
#

if(MULTILIB AND "${TARGET_SYSTEM}" MATCHES "APPLE")
    # On OS X, build "fat" libraries containing code for both architectures.

    # Some suffix for the target/file names of the host-native arch so
    # that they don't collide with the final combined ones.
    set(hostsuffix "${LIB_SUFFIX}${HOST_BITNESS}")

    set(libtargets)
    build_all_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${hostsuffix}" libtargets)
    build_all_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" libtargets)

    # KLUDGE: Cannot use `$<TARGET_LINKER_FILE:target>` in custom command.
    # Set up the list of generated libs (without 'lib' prefix) to be merged manually.
    set(libs_to_install ldc-profile-rt.a)
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        list(APPEND libs_to_install druntime-ldc.a druntime-ldc-debug.a
                                    phobos2-ldc.a  phobos2-ldc-debug.a
        )
    endif()
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        set(suffix ${SHARED_LIB_SUFFIX}.dylib)
        list(APPEND libs_to_install druntime-ldc${suffix} druntime-ldc-debug${suffix}
                                    phobos2-ldc${suffix}  phobos2-ldc-debug${suffix}
        )
    endif()

    foreach(lib ${libs_to_install})
        set(host_path  ${CMAKE_BINARY_DIR}/lib${hostsuffix}/lib${lib})
        set(multi_path ${CMAKE_BINARY_DIR}/lib${MULTILIB_SUFFIX}/lib${lib})
        set(final_path ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/lib${lib})

        add_custom_command(
            OUTPUT  ${final_path}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}
            COMMAND "lipo"
            ARGS    ${multi_path} ${host_path} -create -output ${final_path}
            DEPENDS ${libtargets}
        )

        set(target_name "")
        get_filename_component(target_name ${lib} NAME_WE)
        add_custom_target(${target_name} ALL DEPENDS ${final_path})

        install(FILES       ${final_path}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
    endforeach()
else()
    set(libs_to_install)
    build_all_runtime_variants("" "${RT_CFLAGS}" "${LD_FLAGS}" "${LIB_SUFFIX}" libs_to_install)

    # don't add multilib targets to libs_to_install
    if(MULTILIB)
        build_all_runtime_variants("-m${MULTILIB_SUFFIX}" "-m${MULTILIB_SUFFIX} ${RT_CFLAGS}" "-m${MULTILIB_SUFFIX} ${LD_FLAGS}" "${MULTILIB_SUFFIX}" dummy)
    endif()

    foreach(libname ${libs_to_install})
        install(TARGETS     ${libname}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
        if(MULTILIB)
            install(TARGETS     ${libname}_${MULTILIB_SUFFIX}
                    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${MULTILIB_SUFFIX})
        endif()
    endforeach()
endif()

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


#
# Test targets.
#

# Build the "test runner" executables containing the druntime and Phobos unit
# tests. They are invoked with the modules to test later.
# We just build another copy of the two libraries with -unittest enabled and
# link the test runners against those. Some linker command-line magic is
# required to make sure all objects are pulled in.

macro(append_testrunner_linkflags libname path_suffix is_shared output_flags)
    # (Re)set variable 'tested_lib_path', initialize to directory containing the tested lib.
    # It'll contain the full path to the linked-in library when 'returning' to calling build_test_runners().
    set(tested_lib_path "${CMAKE_BINARY_DIR}/lib${path_suffix}")

    if("${TARGET_SYSTEM}" MATCHES "MSVC")
        set(tested_lib_path "${tested_lib_path}/${libname}.lib")
        if("${is_shared}" STREQUAL "ON")
            append("${tested_lib_path}" ${output_flags})
            append("msvcrt.lib vcruntime.lib" ${output_flags})
        else()
            # the MS linker supports /WHOLEARCHIVE since VS 2015 Update 2
            append("/WHOLEARCHIVE:${tested_lib_path}" ${output_flags})
            append("libcmt.lib libvcruntime.lib" ${output_flags})
        endif()
    else()
        if("${is_shared}" STREQUAL "ON")
            append("-Wl,-rpath,${tested_lib_path}" ${output_flags})
            if("${TARGET_SYSTEM}" MATCHES "APPLE")
                set(tested_lib_path "${tested_lib_path}/lib${libname}.dylib")
                append("${tested_lib_path}" ${output_flags})
            else()
                set(tested_lib_path "${tested_lib_path}/lib${libname}.so")
                append("-Wl,--no-as-needed,${tested_lib_path},--as-needed" ${output_flags})
            endif()
        else()
            set(tested_lib_path "${tested_lib_path}/lib${libname}.a")
            if("${TARGET_SYSTEM}" MATCHES "APPLE")
                append("-Wl,-force_load,${tested_lib_path}" ${output_flags})
            else()
                append("-Wl,--whole-archive,${tested_lib_path},--no-whole-archive" ${output_flags})
            endif()
            foreach(l ${C_SYSTEM_LIBS})
                append("-l${l}" ${output_flags})
            endforeach()
        endif()
    endif()
endmacro()

set(_GLOBAL_TESTRUNNERS "" CACHE INTERNAL "List of all test runner build targets")
file(WRITE ${PROJECT_BINARY_DIR}/dummy.c "")

# Generates targets for a pair of druntime/Phobos test runners.
# The build targets are also appended to the _GLOBAL_TESTRUNNERS list.
function(build_test_runners name_suffix path_suffix d_flags c_flags linkflags is_shared)
    set(target_suffix)
    get_target_suffix("${name_suffix}" "${path_suffix}" target_suffix)

    set(test_runner_o  "")
    set(test_runner_bc "")
    dc("${RUNTIME_DIR}/src/test_runner.d"
       "${RUNTIME_DIR}/src"
       "${d_flags}"
       "${PROJECT_BINARY_DIR}/objects-unittest${target_suffix}"
       "OFF"
       test_runner_o
       test_runner_bc
    )
    add_custom_target(test_runner${target_suffix} DEPENDS ${test_runner_o})

    set(druntime_name druntime-test-runner${target_suffix})
    add_executable(${druntime_name} EXCLUDE_FROM_ALL ${PROJECT_BINARY_DIR}/dummy.c)
    set(full_linkflags "${test_runner_o} ${linkflags}")
    append_testrunner_linkflags("druntime-ldc-unittest${name_suffix}" "${path_suffix}" "${is_shared}" full_linkflags)
    add_dependencies(${druntime_name} test_runner${target_suffix} druntime-ldc-unittest${target_suffix})
    set_target_properties(${druntime_name} PROPERTIES
        COMPILE_FLAGS   ${c_flags}
        LINK_FLAGS      ${full_linkflags}
        LINK_DEPENDS    "${test_runner_o};${tested_lib_path}"
    )
    add_test(build-${druntime_name} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${druntime_name})
    set(_GLOBAL_TESTRUNNERS "${_GLOBAL_TESTRUNNERS};${druntime_name}" CACHE INTERNAL "")

    if(PHOBOS2_DIR)
        set(phobos_name phobos2-test-runner${target_suffix})
        add_executable(${phobos_name} EXCLUDE_FROM_ALL ${PROJECT_BINARY_DIR}/dummy.c)
        target_link_libraries(${phobos_name} druntime-ldc-unittest${target_suffix})
        set(full_linkflags "${test_runner_o} ${linkflags}")
        append_testrunner_linkflags("phobos2-ldc-unittest${name_suffix}" "${path_suffix}" "${is_shared}" full_linkflags)
        add_dependencies(${phobos_name} test_runner${target_suffix} phobos2-ldc-unittest${target_suffix})
        set_target_properties(${phobos_name} PROPERTIES
            COMPILE_FLAGS   ${c_flags}
            LINK_FLAGS      ${full_linkflags}
            LINK_DEPENDS    "${test_runner_o};${tested_lib_path}"
        )
        add_test(build-${phobos_name} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${phobos_name})
        set(_GLOBAL_TESTRUNNERS "${_GLOBAL_TESTRUNNERS};${phobos_name}" CACHE INTERNAL "")
    endif()
endfunction()

# Generates targets for static and/or shared pairs of druntime/Phobos test runners.
function(build_test_runner_variant name_suffix path_suffix d_flags c_flags)
    set(full_d_flags ${D_FLAGS} ${d_flags} -unittest)
    set(full_c_flags "${RT_CFLAGS} ${c_flags}")
    set(linkflags "${LD_FLAGS} ${c_flags}")
    if("${TARGET_SYSTEM}" MATCHES "MSVC")
        # enforce identical COMDAT folding, apparently needed for std.variant
        set(linkflags "/OPT:ICF ${linkflags}")
    endif()

    # static and/or shared pair(s) of druntime/Phobos unittest libs
    set(unittest_libs "")
    build_runtime_variant(
        "${full_d_flags}"
        "${full_c_flags}"
        "${linkflags}"
        "-unittest${name_suffix}"
        "${path_suffix}"
        "OFF" # compile separately
        unittest_libs
    )

    # Only build the unittest libraries when running the tests.
    set_target_properties(${unittest_libs} PROPERTIES EXCLUDE_FROM_ALL ON EXCLUDE_FROM_DEFAULT_BUILD ON)

    # static druntime/Phobos test runners
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        build_test_runners("${name_suffix}" "${path_suffix}" "${full_d_flags}" "${full_c_flags}" "${linkflags}" "OFF")
    endif()
    # shared druntime/Phobos test runners
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        build_test_runners("${name_suffix}${SHARED_LIB_SUFFIX}" "${path_suffix}" "${full_d_flags}" "${full_c_flags}" "${linkflags}" "ON")
    endif()
endfunction()

# Generates targets for static and/or shared debug+release druntime/Phobos test runners.
function(build_test_runner_variants path_suffix d_flags c_flags)
    build_test_runner_variant("" "${path_suffix}" "${D_FLAGS_RELEASE};${d_flags}" "${c_flags}")
    build_test_runner_variant("-debug" "${path_suffix}" "${D_FLAGS_DEBUG};${d_flags}" "${c_flags}")
endfunction()

# Now generate all test runner targets.
build_test_runner_variants("${LIB_SUFFIX}" "" "")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    build_test_runner_variants("${MULTILIB_SUFFIX}" "-m32" "-m32")
endif()

# Let's add a meta build target for all test runners.
add_custom_target(all-test-runners DEPENDS ${_GLOBAL_TESTRUNNERS})

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

function(add_runtime_tests_variants path_suffix)
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "ON")
        add_runtime_tests("" "${path_suffix}")
        add_runtime_tests("-debug" "${path_suffix}")
    endif()
    if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
        add_runtime_tests("${SHARED_LIB_SUFFIX}" "${path_suffix}")
        add_runtime_tests("-debug${SHARED_LIB_SUFFIX}" "${path_suffix}")
    endif()
endfunction()

add_runtime_tests_variants("${LIB_SUFFIX}")
if(MULTILIB AND ${HOST_BITNESS} EQUAL 64)
    add_runtime_tests_variants("${MULTILIB_SUFFIX}")
endif()

# Add the standalone druntime tests.
# TODO: Add test/excetions and test/init_fini.
if(NOT ${BUILD_SHARED_LIBS} STREQUAL "OFF")
    if(MULTILIB AND "${TARGET_SYSTEM}" MATCHES "APPLE")
        # KLUDGE: The library target is a custom command for multilib builds (lipo),
        # so cannot use TARGET_FILE directly. Should stash away that name instead.
        set(druntime_path "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/libdruntime-ldc${SHARED_LIB_SUFFIX}.dylib")
    else()
        set(druntime_path "$<TARGET_FILE:druntime-ldc${SHARED_LIB_SUFFIX}>")
    endif()
    set(outdir ${PROJECT_BINARY_DIR}/druntime-test-shared)

    add_test(NAME clean-druntime-test-shared
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${outdir})

    if("${TARGET_SYSTEM}" MATCHES "FreeBSD")
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
