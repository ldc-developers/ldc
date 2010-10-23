project(ldc)

cmake_minimum_required(VERSION 2.6)
# actually, 2.4 will work, with a warning
# if we required 2.4, 2.6+ would switch to a compatibility mode and cause problems on 2.6+ specifics

include(FindPerl)
if(NOT PERL)
	message(FATAL_ERROR "perl not found")
endif(NOT PERL)

include(FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "pkg-config not found")
else(NOT PKG_CONFIG_FOUND)
    pkg_search_module(LIBCONFIGPP libconfig++)
    if(NOT LIBCONFIGPP_FOUND)
        set(LIBCONFIG_CXXFLAGS "" CACHE STRING "libconfig++ compiler flags")
        set(LIBCONFIG_LDFLAGS "" CACHE STRING "libconfig++ linker flags")
    else(NOT LIBCONFIGPP_FOUND)
        set(LIBCONFIG_CXXFLAGS ${LIBCONFIGPP_CFLAGS} CACHE STRING "libconfig++ compiler flags")
        set(LIBCONFIG_LDFLAGS ${LIBCONFIGPP_LDFLAGS} CACHE STRING "libconfig++ linker flags")
    endif(NOT LIBCONFIGPP_FOUND)
endif(NOT PKG_CONFIG_FOUND)

find_program(LLVM_CONFIG llvm-config ${LLVM_INSTDIR}/bin DOC "path to llvm-config tool")
# get llvm's install dir. a little hackish, we could do something like llvm-config --prefix, but this does as well
string(REPLACE "/bin/llvm-config" "" LLVM_DIR ${LLVM_CONFIG})

set(LLVM_INSTDIR ${LLVM_DIR} CACHE PATH "LLVM installation directory" FORCE)
if(NOT LLVM_INSTDIR)
	message(FATAL_ERROR "llvm not found")
endif(NOT LLVM_INSTDIR)

execute_process(
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --host-target
	OUTPUT_VARIABLE HOST_TARGET
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND /bin/sh ${PROJECT_SOURCE_DIR}/find-alt-triple.sh ${HOST_TARGET}
    OUTPUT_VARIABLE HOST_ALT_TARGET
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
	COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --includedir
	OUTPUT_VARIABLE LLVM_INCLUDEDIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(D_VERSION 1 CACHE STRING "D language version")
set(PROGRAM_PREFIX CACHE STRING "prepended to ldc binary name")
set(PROGRAM_SUFFIX CACHE STRING "appended to ldc binary name")
option(USE_BOEHM_GC "use the Boehm garbage collector internally")
option(GENERATE_OFFTI "generate complete ClassInfo.offTi arrays")
option(USE_METADATA "use metadata and related custom optimization passes")

if(D_VERSION EQUAL 1)
	set(DMDFE_PATH dmd)
	set(LDC_EXE ldc)
	set(LDMD_EXE ldmd)
	add_definitions(-DDMDV1)
elseif(D_VERSION EQUAL 2)
	set(DMDFE_PATH dmd2)
	set(LDC_EXE ldc2)
	set(LDMD_EXE ldmd2)
	add_definitions(-DDMDV2)
else(D_VERSION EQUAL 1)
	message(FATAL_ERROR "unsupported D version")
endif(D_VERSION EQUAL 1)

file(MAKE_DIRECTORY
	${PROJECT_BINARY_DIR}/bin
	${PROJECT_BINARY_DIR}/${DMDFE_PATH}
	${PROJECT_BINARY_DIR}/gen
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
	RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
	COMPILE_FLAGS ${LLVM_CXXFLAGS}
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
set(DEFAULT_ALT_TARGET ${HOST_ALT_TARGET} CACHE STRING "default alt target")

include_directories(. ${DMDFE_PATH} ${DMDFE_PATH}/root ${PROJECT_BINARY_DIR}/${DMDFE_PATH} ${PROJECT_BINARY_DIR} ${LLVM_INCLUDEDIR})

set(EXTRA_LLVM_MODULES "" CACHE STRING "extra llvm components to link in (see llvm-config --components)")
separate_arguments(EXTRA_LLVM_MODULES)
execute_process(
        COMMAND ${PERL_EXECUTABLE} ${LLVM_CONFIG} --libfiles bitwriter linker ipo instrumentation backend ${EXTRA_LLVM_MODULES}
        OUTPUT_VARIABLE LLVM_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# build a define that contains all LLVM targets required and is usable for
# preprocessor code generation. start with the native target.
find_path(LLVM_CONFIG_FILE_PATH config.h PATHS ${LLVM_INCLUDEDIR}/llvm/Config ${LLVM_INCLUDEDIR}/Config NO_DEFAULT_PATH)
if(LLVM_CONFIG_FILE_PATH STREQUAL "LLVM_CONFIG_FILE_PATH-NOTFOUND")
        message("Couldn't find your llvm Config.h file in ${LLVM_INCLUDEDIR}, no native target will be initialized.")
else(LLVM_CONFIG_FILE_PATH STREQUAL "LLVM_CONFIG_FILE_PATH-NOTFOUND")
        file(STRINGS ${LLVM_CONFIG_FILE_PATH}/config.h LLVM_NATIVE_ARCH REGEX "^#define LLVM_NATIVE_ARCH")
        if(LLVM_NATIVE_ARCH)
                string(REGEX REPLACE "^#define LLVM_NATIVE_ARCH (.*)(Target|)$" "\\1" LLVM_NATIVE_ARCH ${LLVM_NATIVE_ARCH})
                message(STATUS "Found native target ${LLVM_NATIVE_ARCH}")
                set(LLVM_MODULES_DEFINE "LLVM_TARGET(${LLVM_NATIVE_ARCH})")
        else(LLVM_NATIVE_ARCH)
                message("Couldn't find the LLVM_NATIVE_ARCH define in ${LLVM_CONFIG_FILE_PATH}/config.h. Probably you have an older LLVM and can ignore this warning.")
        endif(LLVM_NATIVE_ARCH)
endif(LLVM_CONFIG_FILE_PATH STREQUAL "LLVM_CONFIG_FILE_PATH-NOTFOUND")
# chain the extra target list to the define
foreach(EXTRA_TARGET ${EXTRA_LLVM_MODULES})
    set(LLVM_MODULES_DEFINE "${LLVM_MODULES_DEFINE} LLVM_TARGET(${EXTRA_TARGET})")
endforeach(EXTRA_TARGET)
set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/gen/main.cpp PROPERTIES
    COMPILE_DEFINITIONS LDC_TARGETS=${LLVM_MODULES_DEFINE}
)


file(GLOB FE_SRC ${DMDFE_PATH}/*.c)
file(GLOB FE_SRC_ROOT ${DMDFE_PATH}/root/*.c)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
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
	${FE_SRC_ROOT}
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
    -DDISABLE_DEBUG_INFO
)

if(UNIX)
	add_definitions(-DPOSIX)
	set(CONF_INST_DIR bin)
endif(UNIX)

if(USE_BOEHM_GC)
	add_definitions(-DREDIRECT_MALLOC=GC_malloc -DIGNORE_FREE)
endif(USE_BOEHM_GC)

if(GENERATE_OFFTI)
	add_definitions(-DGENERATE_OFFTI)
endif(GENERATE_OFFTI)

if(USE_METADATA)
    add_definitions(-DUSE_METADATA)
endif(USE_METADATA)

if(CMAKE_MINOR_VERSION LESS 6)
	set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "output dir for built executables")
	set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "output dir for built libraries")
	add_definitions(-DDEFAULT_TARGET_TRIPLE=\\"${DEFAULT_TARGET}\\")
    add_definitions(-DDEFAULT_ALT_TARGET_TRIPLE=\\"${DEFAULT_ALT_TARGET}\\")
    add_definitions(-DLDC_INSTALL_PREFIX=\\"${CMAKE_INSTALL_PREFIX}\\")
else(CMAKE_MINOR_VERSION LESS 6)
	add_definitions(-DDEFAULT_TARGET_TRIPLE="${DEFAULT_TARGET}")
    add_definitions(-DDEFAULT_ALT_TARGET_TRIPLE="${DEFAULT_ALT_TARGET}")
    add_definitions(-DLDC_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
endif(CMAKE_MINOR_VERSION LESS 6)

add_executable(${LDC_EXE} ${LDC_SOURCE_FILES})

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})

set_target_properties(
	${LDC_EXE} PROPERTIES
	OUTPUT_NAME ${LDC_EXE_NAME}
	RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
	COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LIBCONFIG_CXXFLAGS} -Wno-deprecated -Wno-write-strings -fexceptions"
)

# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_EXE} "${LLVM_LDFLAGS} ${LLVM_LIBS}" ${LIBCONFIG_LDFLAGS})
if(WIN32)
	target_link_libraries(${LDC_EXE} config++ imagehlp psapi)
	set(CONF_INST_DIR bin)
endif(WIN32)

if(USE_BOEHM_GC)
	target_link_libraries(${LDC_EXE} ${PROJECT_SOURCE_DIR}/libgc.a)
endif(USE_BOEHM_GC)

get_target_property(LDC_LOC ${LDC_EXE} LOCATION)

# TODO: testrun

install(TARGETS ${LDC_EXE} DESTINATION bin)
install(FILES ${PROJECT_SOURCE_DIR}/bin/${LDMD_EXE} DESTINATION bin)
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf DESTINATION ${CONF_INST_DIR})
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.rebuild.conf DESTINATION ${CONF_INST_DIR})
install(DIRECTORY ${PROJECT_BINARY_DIR}/lib DESTINATION . USE_SOURCE_PERMISSIONS)

add_subdirectory(runtime EXCLUDE_FROM_ALL)
