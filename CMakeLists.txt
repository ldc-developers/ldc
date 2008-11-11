project(ldc)

cmake_minimum_required(VERSION 2.6)
# actually, 2.4 will work, with a warning
# if we required 2.4, 2.6+ would switch to a compatibility mode and cause problems on 2.6+ specifics

include(FindPerl)
if(NOT PERL)
	message(FATAL_ERROR "perl not found")
endif(NOT PERL)

set(LLVM_INSTDIR CACHE PATH "LLVM installation directory")

find_program(LLVM_CONFIG llvm-config ${LLVM_INSTDIR}/bin DOC "path to llvm-config tool")
if(NOT LLVM_CONFIG)
	message(FATAL_ERROR "llvm-config not found")
endif(NOT LLVM_CONFIG)

execute_process(
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --host-target
	OUTPUT_VARIABLE HOST_TARGET
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --cxxflags
	OUTPUT_VARIABLE LLVM_CXXFLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --ldflags
	OUTPUT_VARIABLE LLVM_LDFLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --libfiles bitwriter linker ipo instrumentation backend
	OUTPUT_VARIABLE LLVM_LIBS
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
# get llvm's install dir. a little hackish, we could do something like llvm-config --prefix, but this does as well
string(REPLACE "/bin/llvm-config" "" LLVM_INSTDIR ${LLVM_CONFIG})

set(D_VERSION 1 CACHE STRING "D language version")
if(D_VERSION EQUAL 1)
	set(DMDFE_PATH dmd)
	set(LDC_EXE ldc)
	add_definitions(-DDMDV1)
elseif(D_VERSION EQUAL 2)
	set(DMDFE_PATH dmd2)
	set(LDC_EXE ldc2)
	add_definitions(-DDMDV2)
else(D_VERSION EQUAL 1)
	message(FATAL_ERROR "unsupported D version")
endif(D_VERSION EQUAL 1)

file(MAKE_DIRECTORY
	${PROJECT_BINARY_DIR}/bin
	${PROJECT_BINARY_DIR}/${DMDFE_PATH}
)

# idgen and impcnvgen
set_source_files_properties(
	${DMDFE_PATH}/idgen.c
	${DMDFE_PATH}/impcnvgen.c
	PROPERTIES LANGUAGE CXX
)
add_executable(idgen ${DMDFE_PATH}/idgen.c)
add_executable(impcnvgen ${DMDFE_PATH}/impcnvgen.c)
# cmake 2.4
set_target_properties(
	idgen impcnvgen PROPERTIES
	LINKER_LANGUAGE CXX
)
get_target_property(IDGEN_LOC idgen LOCATION)
get_target_property(IMPCNVGEN_LOC impcnvgen LOCATION)
# 
add_custom_command(
	OUTPUT
		${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.c
		${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.h
	# 2.4
	COMMAND ${IDGEN_LOC}
	#COMMAND idgen
	WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
	DEPENDS idgen
)
add_custom_command(
	OUTPUT ${PROJECT_BINARY_DIR}/${DMDFE_PATH}/impcnvtab.c
	# 2.4
	COMMAND ${IMPCNVGEN_LOC}
	#COMMAND impcnvgen 
	WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
	DEPENDS impcnvgen
)
set(LDC_GENERATED
	${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.c
	${PROJECT_BINARY_DIR}/${DMDFE_PATH}/id.h
	${PROJECT_BINARY_DIR}/${DMDFE_PATH}/impcnvtab.c
)
# idgen and impcnvgen done

set(DEFAULT_TARGET ${HOST_TARGET} CACHE STRING "default target")

include_directories(. ${DMDFE_PATH} ${PROJECT_BINARY_DIR}/${DMDFE_PATH} ${LLVM_INSTDIR}/include)

file(GLOB FE_SRC ${DMDFE_PATH}/*.c)
file(GLOB GEN_SRC gen/*.cpp)
file(GLOB IR_SRC ir/*.cpp)
# exclude idgen and impcnvgen and generated sources, just in case
list(REMOVE_ITEM FE_SRC
	${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/idgen.c
	${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvgen.c
	${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/id.c
	${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvtab.c
)
set(LDC_SOURCE_FILES
	${LDC_GENERATED}
	${FE_SRC}
	${GEN_SRC}
	${IR_SRC}
)
set_source_files_properties(
	${LDC_SOURCE_FILES}	PROPERTIES
	LANGUAGE CXX
)

add_definitions(
	-DIN_LLVM
	-D_DH
	-DOPAQUE_VTBLS
	-DUSE_BOEHM_GC=0
	-DX86_REVERSE_PARAMS
	-DX86_PASS_IN_EAX
)

if(UNIX)
	add_definitions(-DPOSIX)
	#set(CONF_SUFFIX conf)
endif(UNIX)

if(CMAKE_MINOR_VERSION LESS 6)
	add_definitions(-DDEFAULT_TARGET_TRIPLE=\\"${DEFAULT_TARGET}\\")
else(CMAKE_MINOR_VERSION LESS 6)
	add_definitions(-DDEFAULT_TARGET_TRIPLE="${DEFAULT_TARGET}")
endif(CMAKE_MINOR_VERSION LESS 6)

add_executable(${LDC_EXE} ${LDC_SOURCE_FILES})

set_target_properties(
	${LDC_EXE} PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
	COMPILE_FLAGS "${LLVM_CXXFLAGS} -Wno-deprecated -Wno-write-strings"
)

# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_EXE} "${LLVM_LDFLAGS} ${LLVM_LIBS}")
if(WIN32)
	target_link_libraries(${LDC_EXE} psapi)
	#set(CONF_SUFFIX ini)
endif(WIN32)

# cmake pre 2.6 doesn't support the RUNTIME_OUTPUT_DIRECTORY target property
if(CMAKE_MINOR_VERSION LESS 6)
	get_target_property(LDC_LOC ${LDC_EXE} LOCATION)
	add_custom_command(
		TARGET ${LDC_EXE}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${LDC_LOC} ${PROJECT_BINARY_DIR}/bin/
		COMMAND ${CMAKE_COMMAND} -E remove ${LDC_LOC}
	)
endif(CMAKE_MINOR_VERSION LESS 6)

#configure_file(${PROJECT_SOURCE_DIR}/ldc.conf.in ${PROJECT_BINARY_DIR}/bin/ldc.${CONF_SUFFIX})
	
# TODO: runtime build
# possible problems:
#	with an oos-build, how to set up a working environment, with conf/ini and tango in place?
#	patch tango? how to determine if that's already been done?
#	std but compatible runtime build, or shared, which doesn't yet work on every platform
#	
# TODO: install target
# TODO: testrun
