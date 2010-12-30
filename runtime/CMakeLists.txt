project(runtime)

cmake_minimum_required(VERSION 2.6)

option(BUILD_SHARED_LIBS "build the runtime as shared libraries (linux only)")
option(BUILD_BC_LIBS "build the runtime as bytecode libraries")
option(BUILD_SINGLE_LIB "build single runtime library" ON)
set(D_FLAGS -g -w -d CACHE STRING "runtime build flags, separated by ;")

if(BUILD_SHARED_LIBS)
	list(APPEND D_FLAGS -relocation-model=pic)
endif(BUILD_SHARED_LIBS)

# build tango for D1, druntime for D2
if(D_VERSION EQUAL 1)
	set(RUNTIME tango)
elseif(D_VERSION EQUAL 2)
	set(RUNTIME druntime)
else(D_VERSION EQUAL 1)
	message(FATAL_ERROR "set d version to 1 or 2")
endif(D_VERSION EQUAL 1)
get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)
set(RUNTIME_DIR ${PROJECT_PARENT_DIR}/../${RUNTIME} CACHE PATH "runtime source dir")

if(D_VERSION EQUAL 1)
	# copy imports to runtime dir
	set(LDC_IMPORTS)
	macro(imports_file SRCARG)
		get_filename_component(DEST ${SRCARG} NAME)
		set(SRC  ${PROJECT_SOURCE_DIR}/${SRCARG})
		set(DEST ${RUNTIME_DIR}/ldc/${DEST})
		
		list(APPEND LDC_IMPORTS ${DEST})
		add_custom_command(
			OUTPUT  ${DEST}
			DEPENDS ${SRC}
			COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC} ${DEST}
		)
	endmacro(imports_file)
	imports_file(import/ldc/cstdarg.di)
	imports_file(import/ldc/intrinsics.di)
	imports_file(import/ldc/llvmasm.di)
	imports_file(internal/ldc/bitmanip.d)
	imports_file(internal/ldc/vararg.d)
	# library names
	set(RUNTIME_CC tango-cc-tango)
	set(RUNTIME_GC tango-gc-basic)
	set(RUNTIME_DC ldc-runtime)
	set(RUNTIME_AIO tango-user-ldc)
	# set paths to source files, or fill lists directly
	set(RUNTIME_DC_DIR ${PROJECT_SOURCE_DIR}/internal)
	set(RUNTIME_GC_DIR ${RUNTIME_DIR}/lib/gc/basic)
	set(RUNTIME_INCLUDE ${RUNTIME_DC_DIR})
	file(GLOB CORE_D ${RUNTIME_DIR}/lib/common/tango/core/*.d)
	file(GLOB CORE_C ${RUNTIME_DIR}/lib/common/tango/stdc/*.c)
	file(GLOB_RECURSE GC_D ${RUNTIME_GC_DIR}/*.d)
	file(GLOB_RECURSE DCRT_D ${RUNTIME_DC_DIR}/*.d)
	file(GLOB DCRT_C ${RUNTIME_DC_DIR}/*.c)
elseif(D_VERSION EQUAL 2)
	set(RUNTIME_CC druntime-core)
	set(RUNTIME_GC druntime-gc-basic)
	set(RUNTIME_DC druntime-rt-ldc)
	set(RUNTIME_AIO druntime-ldc)
	set(RUNTIME_DC_DIR ${RUNTIME_DIR}/src/rt)
	set(RUNTIME_GC_DIR ${RUNTIME_DIR}/src/gc)
	set(RUNTIME_INCLUDE ${RUNTIME_DIR}/src)
	file(GLOB CORE_D ${RUNTIME_DIR}/src/core/*.d )
	file(GLOB CORE_D_SYNC ${RUNTIME_DIR}/src/core/sync/*.d )
	file(GLOB_RECURSE GC_D ${RUNTIME_GC_DIR}/*.d)
	file(GLOB_RECURSE DCRT_D ${RUNTIME_DC_DIR}/*.d)
	list(REMOVE_ITEM DCRT_D
		${RUNTIME_DC_DIR}/arraybyte.d
		${RUNTIME_DC_DIR}/arraycast.d
		${RUNTIME_DC_DIR}/arraycat.d
		${RUNTIME_DC_DIR}/arraydouble.d
		${RUNTIME_DC_DIR}/arrayfloat.d
		${RUNTIME_DC_DIR}/arrayreal.d
		${RUNTIME_DC_DIR}/arrayshort.d
		${RUNTIME_DC_DIR}/deh2.d
	)
	file(GLOB DCRT_C ${RUNTIME_DC_DIR}/*.c)
	list(REMOVE_ITEM DCRT_C ${RUNTIME_DC_DIR}/deh.c ${RUNTIME_DC_DIR}/memory_osx.c)
	if(UNIX)
		file(GLOB_RECURSE CORE_D_SYS ${RUNTIME_DIR}/src/core/sys/posix/*.d)
	elseif(WIN32)
		file(GLOB_RECURSE CORE_D_SYS ${RUNTIME_DIR}/src/core/sys/windows/*.d)
	elseif(APPLE)
		file(GLOB_RECURSE CORE_D_SYS ${RUNTIME_DIR}/src/core/sys/osx/*.d)
	endif(UNIX)
	list(APPEND CORE_D ${CORE_D_SYNC} ${CORE_D_SYS} ${RUNTIME_DIR}/src/object_.d
		${RUNTIME_DIR}/src/std/intrinsic.d
		${RUNTIME_DIR}/src/core/stdc/stdarg.d
	)
	file(GLOB CORE_C ${RUNTIME_DIR}/src/core/stdc/*.c)

    if(PHOBOS2_DIR)
        file(GLOB PHOBOS2_D ${PHOBOS2_DIR}/std/*.d)
        file(GLOB PHOBOS2_D_MATH ${PHOBOS2_DIR}/std/internal/math/*.d)
        file(GLOB PHOBOS2_D_C ${PHOBOS2_DIR}/std/c/*.d)
        if(UNIX)
            file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/linux/*.d)
        elseif(WIN32)
            file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/windows/*.d)
        elseif(APPLE)
            file(GLOB PHOBOS2_D_C_SYS ${PHOBOS2_DIR}/std/c/osx/*.d)
        endif(UNIX)
        file(GLOB ZLIB_C ${PHOBOS2_DIR}/etc/c/zlib/*.c)
        list(REMOVE_ITEM ZLIB_C
            ${PHOBOS2_DIR}/etc/c/zlib/minigzip.c
            ${PHOBOS2_DIR}/etc/c/zlib/example.c
            ${PHOBOS2_DIR}/etc/c/zlib/gzio.c
        )
        if(WIN32)
            file(GLOB PHOBOS2_D_WIN ${PHOBOS2_DIR}/std/windows/*.d)
        endif(WIN32)
        list(APPEND PHOBOS2_D
                ${PHOBOS2_D_MATH}
                ${PHOBOS2_D_WIN}
                ${PHOBOS2_D_C}
                ${PHOBOS2_D_C_SYS}
                ${PHOBOS2_DIR}/etc/c/zlib.d
                ${PHOBOS2_DIR}/crc32.d
        )
    endif(PHOBOS2_DIR)
endif(D_VERSION EQUAL 1)

# should only be necessary if run independently from ldc cmake project
if(NOT LDC_LOC)
	if(NOT LDC_EXE)
		if(D_VERSION EQUAL 1)
			set(LDC_EXE ldc)
		elseif(D_VERSION EQUAL 2)
			set(LDC_EXE ldc2)
		endif(D_VERSION EQUAL 1)
	endif(NOT LDC_EXE)

	find_program(LDC_LOC ${LDC_EXE} ../bin DOC "path to ldc binary")
	if(NOT LDC_LOC)
		message(SEND_ERROR "ldc not found")
	endif(NOT LDC_LOC)
	set(LDC_EXE_NAME ${LDC_EXE})
endif(NOT LDC_LOC)

configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf)
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.rebuild.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.rebuild.conf)

# patch runtime source, uses LDC_EXE for ldc2
if(D_VERSION EQUAL 2)
	find_program(PATCH_EXE patch DOC "path to patch tool")
	if(NOT PATCH_EXE)
		message(STATUS "patch tool not found, can't automatically patch runtime sources for ldc")
	else(NOT PATCH_EXE)
		add_custom_command(
			OUTPUT patch-runtime
			COMMAND ${PATCH_EXE} -t -N -p0 -i ${PROJECT_SOURCE_DIR}/${LDC_EXE}.diff
			WORKING_DIRECTORY ${RUNTIME_DIR}
		)
		# rebuild cache to include sources added by patch
		add_custom_command(
			OUTPUT recache
			COMMAND ${CMAKE_COMMAND} -H${PROJECT_PARENT_DIR} -B${PROJECT_BINARY_DIR}/..
		)
		add_custom_target(patch DEPENDS patch-runtime recache ${LDC_IMPORTS})
	endif(NOT PATCH_EXE)
endif(D_VERSION EQUAL 2)

macro(dc INPUT_D OUTLIST_O OUTLIST_BC INCDIR MOREFLAGS PATH)
	if ("${PATH}" STREQUAL "")
		file(RELATIVE_PATH output ${RUNTIME_DIR} ${INPUT_D})
	else ("${PATH}" STREQUAL "")
		file(RELATIVE_PATH output ${PATH} ${INPUT_D})
	endif ("${PATH}" STREQUAL "")
	get_filename_component(name ${output} NAME_WE)
	get_filename_component(path ${output} PATH)
	if ("${path}" STREQUAL "")
		set(output ${name})
	else ("${path}" STREQUAL "")
		set(output ${path}/${name})
	endif ("${path}" STREQUAL "")
	set(OUTPUT_O ${PROJECT_BINARY_DIR}/${output}.o)
	set(OUTPUT_BC ${PROJECT_BINARY_DIR}/${output}.bc)
	list(APPEND ${OUTLIST_O} ${OUTPUT_O})
	list(APPEND ${OUTLIST_BC} ${OUTPUT_BC})

	# Compile
	add_custom_command(
		OUTPUT
			${OUTPUT_O}
			${OUTPUT_BC}
				COMMAND ${LDC_LOC} --output-o --output-bc -c -I${INCDIR} -I${RUNTIME_GC_DIR} ${INPUT_D} -of${OUTPUT_O} ${D_FLAGS} ${MOREFLAGS}
		DEPENDS ${LDC_LOC}
			${INPUT_D}
			${LDC_IMPORTS}
			${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}.conf
	)
endmacro(dc)

foreach(f ${CORE_D})
	dc(${f} CORE_O CORE_BC ${RUNTIME_INCLUDE} "-disable-invariants" "")
endforeach(f)

foreach(f ${GC_D})
	dc(${f} GC_O GC_BC ${RUNTIME_INCLUDE} "-disable-invariants" "")
endforeach(f)

foreach(f ${DCRT_D})
	if(D_VERSION EQUAL 1)
		dc(${f} DCRT_O DCRT_BC ${RUNTIME_INCLUDE} "" ${RUNTIME_DC_DIR})
	else(D_VERSION EQUAL 1)
		dc(${f} DCRT_O DCRT_BC ${RUNTIME_INCLUDE} "" "")
	endif(D_VERSION EQUAL 1)
endforeach(f)

if(BUILD_SINGLE_LIB)
	add_library(${RUNTIME_AIO} ${CORE_O} ${CORE_C} ${GC_O} ${DCRT_O} ${DCRT_C})
	set(LIBS ${RUNTIME_AIO})
else(BUILD_SINGLE_LIB)
	add_library(${RUNTIME_CC} ${CORE_O} ${CORE_C})
	add_library(${RUNTIME_GC} ${GC_O})
	add_library(${RUNTIME_DC} ${DCRT_O} ${DCRT_C})
	set(LIBS
		${RUNTIME_CC}
		${RUNTIME_GC}
		${RUNTIME_DC}
	)
endif(BUILD_SINGLE_LIB)

if(BUILD_BC_LIBS)
	find_program(LLVM_AR_EXE llvm-ar ${LLVM_INSTDIR}/bin DOC "path to llvm-ar tool")
	if(NOT LLVM_AR_EXE)
		message(SEND_ERROR "llvm-ar not found")
	endif(NOT LLVM_AR_EXE)

	add_library(${RUNTIME_CC}-c ${CORE_C})
	add_library(${RUNTIME_DC}-c ${DCRT_C})
	list(APPEND LIBS
		${RUNTIME_CC}-c
		${RUNTIME_DC}-c
	)
	add_custom_command(
		OUTPUT bclibs
		COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_CC}-bc.a ${CORE_BC}
		COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_GC}-bc.a ${GC_BC}
		# cannot parse genobj.bc if built with -g
		# COMMAND ${LLVM_AR_EXE} rs lib${RUNTIME_DC}-bc.a ${DCRT_BC}
		WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
		DEPENDS
			${CORE_BC}
			${GC_BC}
			${DCRT_BC}
			${LDC_IMPORTS}
	)
	set(BCLIBS bclibs)
endif(BUILD_BC_LIBS)

set_target_properties(
    ${LIBS} PROPERTIES
    LINKER_LANGUAGE C
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
    COMPILE_FLAGS "-m32"
    LINK_FLAGS "-m32"
)

# BCLIBS is empty if BUILD_BC_LIBS is not selected
add_custom_target(runtime DEPENDS ${LIBS} ${BCLIBS})

if(PHOBOS2_DIR)
    foreach(f ${PHOBOS2_D})
         dc(${f} PHOBOS2_O PHOBOS2_BC ${RUNTIME_DIR}/src/ "-I${PHOBOS2_DIR}" ${PHOBOS2_DIR})
    endforeach(f)

    add_library(phobos2 ${ZLIB_C} ${PHOBOS2_O})
    add_dependencies(phobos2 runtime)
    set_target_properties(
        phobos2 PROPERTIES
        LINKER_LANGUAGE C
        ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
        COMPILE_FLAGS "-m32"
        LINK_FLAGS "-m32"
    )
endif(PHOBOS2_DIR)
