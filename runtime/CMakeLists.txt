project(runtime)

cmake_minimum_required(VERSION 2.6)

find_program(GCC_EXE gcc DOC "path to gcc binary")
if(NOT GCC_EXE)
	message(STATUS "gcc needs to be on your path to build the runtime")
endif(NOT GCC_EXE)

option(BUILD_SHARED_LIBS "build the runtime as shared libraries (linux only)")
option(BUILD_BC_LIBS "build the runtime as bytecode libraries")
option(BUILD_SINGLE_LIB "build single runtime library" ON)
set(D_FLAGS -g -w -d CACHE STRING "runtime build flags, separated by ;")

if(BUILD_SHARED_LIBS)
	list(APPEND D_FLAGS -fPIC)
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
set(RUNTIME_DIR ${PROJECT_PARENT_DIR}/${RUNTIME} CACHE PATH "runtime source dir" FORCE)

if(D_VERSION EQUAL 1)
	# copy imports to runtime dir
	add_custom_command(
		OUTPUT ldc-imports
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/import/ldc ${RUNTIME_DIR}/ldc
		COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/internal/ldc/bitmanip.d ${RUNTIME_DIR}/ldc
		COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/internal/ldc/vararg.d ${RUNTIME_DIR}/ldc
	)
	set(LDC_IMPORTS ldc-imports)
	# library names
	set(RUNTIME_CC tango-cc-tango)
	set(RUNTIME_GC tango-gc-basic)
	set(RUNTIME_DC ldc-runtime)
	set(RUNTIME_AIO tango-ldc)
	# set paths to source files, or fill lists directly
	set(RUNTIME_DC_DIR ${PROJECT_SOURCE_DIR}/internal)
	set(RUNTIME_GC_DIR ${RUNTIME_DIR}/lib/gc/basic)
	file(GLOB CORE_D ${RUNTIME_DIR}/lib/common/tango/core/*.d)
	file(GLOB CORE_C ${RUNTIME_DIR}/lib/common/tango/stdc/*.c)
elseif(D_VERSION EQUAL 2)
	set(RUNTIME_CC druntime-core)
	set(RUNTIME_GC druntime-gc-basic)
	set(RUNTIME_DC druntime-rt-ldc)
	set(RUNTIME_AIO druntime-ldc)
	set(RUNTIME_DC_DIR ${RUNTIME_DIR}/src/compiler/ldc)
	set(RUNTIME_GC_DIR ${RUNTIME_DIR}/src/gc/basic)
	file(GLOB CORE_D ${RUNTIME_DIR}/src/common/core/*.d)
	file(GLOB CORE_C ${RUNTIME_DIR}/src/common/core/stdc/*.c)
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

configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE_NAME}.conf)

# patch runtime source, uses LDC_EXE for ldc / ldc2
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

file(GLOB GC_D ${RUNTIME_GC_DIR}/*.d)
file(GLOB_RECURSE DCRT_D ${RUNTIME_DC_DIR}/*.d)
file(GLOB DCRT_C ${RUNTIME_DC_DIR}/*.c)

# compile d file into outdir, include incdir, and append names of generated .o and .bc to outlist_o and _bc
macro(dc INPUT_D OUTLIST_O OUTLIST_BC OUTDIR INCDIR MOREFLAGS)
	get_filename_component(BASENAME ${INPUT_D} NAME_WE)
	set(OUTPUT_O ${PROJECT_BINARY_DIR}/${OUTDIR}/${BASENAME}.o)
	set(OUTPUT_BC ${PROJECT_BINARY_DIR}/${OUTDIR}/${BASENAME}.bc)
	list(APPEND ${OUTLIST_O} ${OUTPUT_O})
	list(APPEND ${OUTLIST_BC} ${OUTPUT_BC})
	add_custom_command(
		OUTPUT
			${OUTPUT_O}
			${OUTPUT_BC}
		COMMAND ${LDC_LOC} -c -I${INCDIR} -output-bc ${INPUT_D} -of${OUTPUT_O} ${D_FLAGS} ${MOREFLAGS}
		DEPENDS ${LDC_LOC}
	)
endmacro(dc)

# dc_dir include for core and gc only necessary with druntime
foreach(f ${CORE_D})
	dc(${f} CORE_O CORE_BC core ${RUNTIME_DC_DIR} "")
endforeach(f)

foreach(f ${GC_D})
	dc(${f} GC_O GC_BC gc "${RUNTIME_GC_DIR} ${RUNTIME_DC_DIR}" "-disable-invariants")
endforeach(f)

foreach(f ${DCRT_D})
	dc(${f} DCRT_O DCRT_BC dcrt ${RUNTIME_DC_DIR} "")
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
	)
	set(BCLIBS bclibs)
endif(BUILD_BC_LIBS)

set_target_properties(
	${LIBS} PROPERTIES
	LINKER_LANGUAGE C
	ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
	LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib
)

# BCLIBS is empty if BUILD_BC_LIBS is not selected
add_custom_target(runtime DEPENDS ${LIBS} ${BCLIBS})

install(DIRECTORY ${PROJECT_BINARY_DIR}/../lib DESTINATION .)