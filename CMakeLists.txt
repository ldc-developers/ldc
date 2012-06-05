project(ldc)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

#
# Locate LLVM.
#

# This would actually better be named named EXTRA_LLVM_TARGETS, as it allows
# additional targets (beside the native one) to be specified. It affects the
# LLVM libraries linked and is converted to a preprocessor define used in
# gen/main.cpp.
set(EXTRA_LLVM_MODULES "" CACHE STRING
    "Extra LLVM targets to link in (see llvm-config --targets-built)")
separate_arguments(EXTRA_LLVM_MODULES)

# We need to find exactly the right LLVM version, our code usually does not
# work across LLVM »minor« releases.
find_package(LLVM 3.0 EXACT REQUIRED
    bitwriter linker ipo instrumentation backend ${EXTRA_LLVM_MODULES})

#
# Locate libconfig++.
#

include(FindPkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_search_module(LIBCONFIGPP libconfig++)
    if(LIBCONFIGPP_FOUND)
        set(LIBCONFIG_CXXFLAGS ${LIBCONFIGPP_CFLAGS} CACHE STRING "libconfig++ compiler flags")
        set(LIBCONFIG_LDFLAGS ${LIBCONFIGPP_LDFLAGS} CACHE STRING "libconfig++ linker flags")
    endif()
else()
    message(WARNING "pkg-config not found. libconfig++ options cannot be detected, you might need to set them manually.")
endif()

# libconfig++ is actually a required dependency, but has never been defined as
# one – maybe there was a problem with the auto detection once.
if (NOT LIBCONFIGPP_FOUND)
    set(LIBCONFIG_CXXFLAGS "" CACHE STRING "libconfig++ compiler flags")
    set(LIBCONFIG_LDFLAGS "" CACHE STRING "libconfig++ linker flags")
    message(WARNING "libconfig++ not found. If compilation fails but the library is installed, consider manually setting the LIBCONFIG_CXXFLAGS and LIBCONFIG_LDFLAGS CMake variables.")
endif()

#
# Main configuration.
#

# Generally, we want to install everything into CMAKE_INSTALL_PREFIX, but when
# it is /usr, put the config files into /etc to meet common practice.
if (NOT DEFINED SYSCONF_INSTALL_DIR)
    if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
        set(SYSCONF_INSTALL_DIR "/etc")
    else(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
        set(SYSCONF_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/etc")
    endif(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
endif (NOT DEFINED SYSCONF_INSTALL_DIR)

set(D_VERSION 2 CACHE STRING "D language version")
set(PROGRAM_PREFIX "" CACHE STRING "Prepended to ldc/ldmd binary names")
set(PROGRAM_SUFFIX "" CACHE STRING "Appended to ldc/ldmd binary names")
set(CONF_INST_DIR ${SYSCONF_INSTALL_DIR} CACHE PATH "Directory ldc.conf is installed to")

# The following flags are currently not well tested, expect the build to fail.
option(USE_BOEHM_GC "use the Boehm garbage collector internally")
option(GENERATE_OFFTI "generate complete ClassInfo.offTi arrays")
option(USE_METADATA "use metadata and related custom optimization passes")
mark_as_advanced(USE_BOEHM_GC GENERATE_OFFTI USE_METADATA)

if(D_VERSION EQUAL 1)
    set(DMDFE_PATH dmd)
    set(LDC_EXE ldc)
    set(LDMD_EXE ldmd)
    set(RUNTIME runtime)
    add_definitions(-DDMDV1)
elseif(D_VERSION EQUAL 2)
    set(DMDFE_PATH dmd2)
    set(LDC_EXE ldc2)
    set(LDMD_EXE ldmd2)
    set(RUNTIME druntime)
    add_definitions(-DDMDV2)
else(D_VERSION EQUAL 1)
    message(FATAL_ERROR "unsupported D version")
endif(D_VERSION EQUAL 1)

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})
set(LDMD_EXE_NAME ${PROGRAM_PREFIX}${LDMD_EXE}${PROGRAM_SUFFIX})

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    ${PROJECT_BINARY_DIR}/gen
)

#
# Run idgen and impcnvgen.
#
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

#
# Set up target defines.
#

set(DEFAULT_TARGET ${LLVM_HOST_TARGET} CACHE STRING "default target")
add_definitions(-DDEFAULT_TARGET_TRIPLE="${DEFAULT_TARGET}")

# Generate the alternate target triple (x86 on x86_64 and vice versa.)
execute_process(
    COMMAND /bin/sh ${PROJECT_SOURCE_DIR}/find-alt-triple.sh ${LLVM_HOST_TARGET}
    OUTPUT_VARIABLE HOST_ALT_TARGET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(DEFAULT_ALT_TARGET ${HOST_ALT_TARGET} CACHE STRING "default alt target")
add_definitions(-DDEFAULT_ALT_TARGET_TRIPLE="${DEFAULT_ALT_TARGET}")

#
# Detect host architecture.
# The code borrowed from llvm's config-x.cmake.
#
# This is only needed to initialize the llvm native target which is
# exactly the purpose of llvm::InitializeNativeTarget* functions.
# Unfortunately, there is a bug in llvm's cmake script that prevents
# the asm parser from being initialized when the functions are used.
# So we have to do the dirty work ourselves.
string(REGEX MATCH "^[^-]*" HOST_ARCH ${LLVM_HOST_TARGET})
if(HOST_ARCH MATCHES "i[2-6]86")
  set(HOST_ARCH X86)
elseif(HOST_ARCH STREQUAL "x86")
  set(HOST_ARCH X86)
elseif(HOST_ARCH STREQUAL "amd64")
  set(HOST_ARCH X86)
elseif(HOST_ARCH STREQUAL "x86_64")
  set(HOST_ARCH X86)
elseif(HOST_ARCH MATCHES "sparc")
  set(HOST_ARCH Sparc)
elseif(HOST_ARCH MATCHES "powerpc")
  set(HOST_ARCH PowerPC)
elseif(HOST_ARCH MATCHES "alpha")
  set(HOST_ARCH Alpha)
elseif(HOST_ARCH MATCHES "arm")
  set(HOST_ARCH ARM)
elseif(HOST_ARCH MATCHES "mips")
  set(HOST_ARCH Mips)
elseif(HOST_ARCH MATCHES "xcore")
  set(HOST_ARCH XCore)
elseif(HOST_ARCH MATCHES "msp430")
  set(HOST_ARCH MSP430)
else(HOST_ARCH MATCHES "i[2-6]86")
  message(FATAL_ERROR "Unknown architecture ${HOST_ARCH}")
endif(HOST_ARCH MATCHES "i[2-6]86")

# Pass the list of LLVM targets as preprocessor constants.
foreach(TARGET ${HOST_ARCH} ${EXTRA_LLVM_MODULES})
    set(LLVM_MODULES_DEFINE "${LLVM_MODULES_DEFINE} LLVM_TARGET(${TARGET})")
endforeach(TARGET)

set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/driver/main.cpp PROPERTIES
    COMPILE_DEFINITIONS LDC_TARGETS=${LLVM_MODULES_DEFINE}
)

#
# Gather source files.
#

# Also add the header files to the build so that they are available in IDE
# project files generated via CMake.
file(GLOB_RECURSE FE_SRC ${DMDFE_PATH}/*.c ${DMDFE_PATH}/*.h)
file(GLOB_RECURSE GEN_SRC gen/*.cpp gen/*.h)
file(GLOB IR_SRC ir/*.cpp ir/*.h)
set(DRV_SRC
    driver/cl_options.cpp
    driver/configfile.cpp
    driver/linker.h
    driver/toobj.cpp
    driver/cl_options.h
    driver/configfile.h
    driver/linker.cpp
    driver/main.cpp
    driver/toobj.h
)
# exclude idgen and impcnvgen and generated sources, just in case
list(REMOVE_ITEM FE_SRC
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/idgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/id.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvtab.c
)
# exclude root/win32.c on non-windows systems
if(NOT WIN32)
    list(REMOVE_ITEM FE_SRC ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/root/win32.c)
endif(NOT WIN32)
# disable dmd gc
list(REMOVE_ITEM FE_SRC ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/root/dmgcmem.c)
set(LDC_SOURCE_FILES
    ${LDC_GENERATED}
    ${FE_SRC}
    ${GEN_SRC}
    ${IR_SRC}
)

# DMD source files have a .c extension, but are actually C++ code.
foreach(file ${LDC_SOURCE_FILES})
    if(file MATCHES ".*\\.c$")
        set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
    endif()
endforeach()

#
# Includes, defines.
#

include_directories(
    .
    ${DMDFE_PATH}
    ${DMDFE_PATH}/root
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
)

add_definitions(
    -DIN_LLVM
    -DOPAQUE_VTBLS
    -DLDC_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
)

if(UNIX)
    add_definitions(-DPOSIX)
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

#
# Set up the main ldc/ldc2 target.
#
if(BUILD_SHARED)
    set(LDC_LIB_TYPE SHARED)
else(BUILD_SHARED)
    set(LDC_LIB_TYPE STATIC)
endif(BUILD_SHARED)

set(LDC_LIB LDCShared)
add_library(${LDC_LIB} ${LDC_LIB_TYPE} ${LDC_SOURCE_FILES})
set_target_properties(
    ${LDC_LIB} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_NAME ldc
    LIBRARY_OUTPUT_NAME ldc
    RUNTIME_OUTPUT_NAME ldc
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LIBCONFIG_CXXFLAGS} -Wno-deprecated -Wno-write-strings -fexceptions"
)

# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} "${LLVM_LDFLAGS} ${LLVM_LIBRARIES}")
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif(WIN32)

if(USE_BOEHM_GC)
    target_link_libraries(${LDC_LIB} ${PROJECT_SOURCE_DIR}/libgc.a)
endif(USE_BOEHM_GC)


add_executable(${LDC_EXE} ${DRV_SRC})
set_target_properties(
    ${LDC_EXE} PROPERTIES
    OUTPUT_NAME ${LDC_EXE_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LIBCONFIG_CXXFLAGS} -Wno-deprecated -Wno-write-strings -fexceptions"
)
target_link_libraries(${LDC_EXE} ${LDC_LIB} ${LIBCONFIG_LDFLAGS} config++)

# For use by the druntime/Phobos build system.
get_target_property(LDC_LOC ${LDC_EXE} LOCATION)

#
# LDMD
#
set_source_files_properties(dmd2/root/response.c dmd2/root/man.c PROPERTIES LANGUAGE CXX)
add_executable(${LDMD_EXE} dmd2/root/response.c dmd2/root/man.c driver/ldmd.cpp)
set_target_properties(${LDMD_EXE} PROPERTIES
    COMPILE_DEFINITIONS LDC_EXE_NAME="${LDC_EXE_NAME}"
    COMPILE_FLAGS "${LLVM_CXXFLAGS}"
    OUTPUT_NAME "${LDMD_EXE_NAME}"
)
target_link_libraries(${LDMD_EXE} "${LLVM_LDFLAGS} ${LLVM_LIBRARIES}")

#
# Install target.
#

add_subdirectory(runtime)

install(TARGETS ${LDC_EXE} ${LDMD_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
if(${BUILD_SHARED})
    # For now, only install libldc if explicitely building the shared library.
    # While it might theoretically be possible to use LDC as a static library
    # as well, for the time being this just bloats the normal packages.
    install(TARGETS ${LDC_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}_install.conf DESTINATION ${CONF_INST_DIR} RENAME ${LDC_EXE}.conf)
install(FILES ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}_install.rebuild.conf DESTINATION ${CONF_INST_DIR} RENAME ${LDC_EXE}.rebuild.conf)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    install(DIRECTORY bash_completion.d DESTINATION ${CONF_INST_DIR})
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

