project(ldc)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

if(MSVC)
    set(LIBCONFIG_DLL OFF CACHE BOOL "Use libconfig++ DLL instead of static library")
endif()

#
# Locate LLVM.
#

find_package(LLVM 3.0 REQUIRED
    all-targets bitwriter linker ipo instrumentation backend support tablegen asmparser ${EXTRA_LLVM_MODULES})
math(EXPR LDC_LLVM_VER ${LLVM_VERSION_MAJOR}*100+${LLVM_VERSION_MINOR})

#
# Locate libconfig++.
#
find_package(LibConfig++ REQUIRED)

#
# Main configuration.
#

# Generally, we want to install everything into CMAKE_INSTALL_PREFIX, but when
# it is /usr, put the config files into /etc to meet common practice.
if(NOT DEFINED SYSCONF_INSTALL_DIR)
    if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
        set(SYSCONF_INSTALL_DIR "/etc")
    else()
        set(SYSCONF_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/etc")
    endif()
endif()

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
else()
    message(FATAL_ERROR "unsupported D version")
endif()

set(LDC_EXE_NAME ${PROGRAM_PREFIX}${LDC_EXE}${PROGRAM_SUFFIX})
set(LDMD_EXE_NAME ${PROGRAM_PREFIX}${LDMD_EXE}${PROGRAM_SUFFIX})

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/${DMDFE_PATH}
)

if(MSVC)
    ENABLE_LANGUAGE(ASM_MASM)
endif()

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
    COMPILE_FLAGS "${LLVM_CXXFLAGS}"
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
# Gather source files.
#

# Also add the header files to the build so that they are available in IDE
# project files generated via CMake.
file(GLOB_RECURSE FE_SRC ${DMDFE_PATH}/*.c)
file(GLOB_RECURSE FE_HDR ${DMDFE_PATH}/*.h)
file(GLOB_RECURSE GEN_SRC gen/*.cpp)
file(GLOB_RECURSE GEN_HDR gen/*.h)
file(GLOB IR_SRC ir/*.cpp)
file(GLOB IR_HDR ir/*.h)
set(DRV_SRC
    driver/cl_options.cpp
    driver/configfile.cpp
    driver/toobj.cpp
    driver/linker.cpp
    driver/main.cpp
)
set(DRV_HDR
    driver/linker.h
    driver/cl_options.h
    driver/configfile.h
    driver/toobj.h
)
# exclude idgen and impcnvgen and generated sources, just in case
list(REMOVE_ITEM FE_SRC
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/idgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvgen.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/id.c
    ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/impcnvtab.c
)
# Add/remove files for MSVC
if(MSVC)
    list(REMOVE_ITEM FE_SRC
        ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/root/gnuc.c
    )
    list(REMOVE_ITEM FE_HDR
        ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/root/gnuc.h
    )
    list(APPEND FE_SRC
        ${PROJECT_SOURCE_DIR}/vcbuild/strtold.c
# See below why this don't work
#        if(CMAKE_CL_64)
#        ${PROJECT_SOURCE_DIR}/vcbuild/ldfpu.asm
#        endif()
    )
    if(CMAKE_CL_64)
        # MASM support does not work yet!
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ldfpu.obj
                           COMMAND ${CMAKE_ASM_MASM_COMPILER} /c /Fo${CMAKE_CURRENT_BINARY_DIR}/ldfpu.obj ${PROJECT_SOURCE_DIR}/vcbuild/ldfpu.asm
                           DEPENDS ${PROJECT_SOURCE_DIR}/vcbuild/ldfpu.asm
                           COMMENT "generate ldfpu.obj")
        list(APPEND FE_SRC
            ${CMAKE_CURRENT_BINARY_DIR}/ldfpu.obj
    )
    endif()
endif()
# disable dmd gc
list(REMOVE_ITEM FE_SRC ${PROJECT_SOURCE_DIR}/${DMDFE_PATH}/root/dmgcmem.c)
set(LDC_SOURCE_FILES
    ${LDC_GENERATED}
    ${FE_SRC}
    ${FE_HDR}
    ${GEN_SRC}
    ${GEN_HDR}
    ${IR_SRC}
    ${IR_HDR}
)

# DMD source files have a .c extension, but are actually C++ code.
foreach(file ${LDC_SOURCE_FILES})
    if(file MATCHES ".*\\.c$")
        set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
    endif()
endforeach()

source_group("Source Files\\${DMDFE_PATH}" FILES ${FE_SRC})
source_group("Header Files\\${DMDFE_PATH}" FILES ${FE_HDR})
source_group("Source Files\\gen" FILES ${GEN_SRC})
source_group("Header Files\\gen" FILES ${GEN_HDR})
source_group("Source Files\\ir" FILES ${IR_SRC})
source_group("Header Files\\ir" FILES ${IR_HDR})
source_group("Generated Files" REGULAR_EXPRESSION "(id\\.[ch]|impcnvtab\\.c)$")

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
    ${LIBCONFIG++_INCLUDE_DIR}
)

if(MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/vcbuild)
    if(NOT LIBCONFIG_DLL)
        add_definitions(-DLIBCONFIGXX_STATIC -DLIBCONFIG_STATIC)
    endif()
endif()

add_definitions(
    -DIN_LLVM
    -DOPAQUE_VTBLS
    -DLDC_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    -DLDC_LLVM_VER=${LDC_LLVM_VER}
    -DLDC_LLVM_VERSION_STRING="${LLVM_VERSION_STRING}"
)

if(UNIX)
    add_definitions(-DPOSIX)
endif()

if(USE_BOEHM_GC)
    add_definitions(-DREDIRECT_MALLOC=GC_malloc -DIGNORE_FREE)
endif()

if(GENERATE_OFFTI)
    add_definitions(-DGENERATE_OFFTI)
endif()

if(USE_METADATA)
    add_definitions(-DUSE_METADATA)
endif()

if(MSVC)
    set(EXTRA_CXXFLAGS "/W0 /wd4996 /GF /GR- /RTC1")
else()
    set(EXTRA_CXXFLAGS "-Wno-deprecated -Wno-write-strings -fexceptions")
endif()

#
# Set up the main ldc/ldc2 target.
#
if(BUILD_SHARED)
    set(LDC_LIB_TYPE SHARED)
else()
    set(LDC_LIB_TYPE STATIC)
endif()

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
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${EXTRA_CXXFLAGS}"
)

# LDFLAGS should actually be in target property LINK_FLAGS, but this works, and gets around linking problems
target_link_libraries(${LDC_LIB} "${LLVM_LDFLAGS}" ${LLVM_LIBRARIES})
if(WIN32)
    target_link_libraries(${LDC_LIB} imagehlp psapi)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${LDC_LIB} dl)
endif()

if(USE_BOEHM_GC)
    target_link_libraries(${LDC_LIB} ${PROJECT_SOURCE_DIR}/libgc.a)
endif()


add_executable(${LDC_EXE} ${DRV_SRC} ${DRV_HDR})
set_target_properties(
    ${LDC_EXE} PROPERTIES
    OUTPUT_NAME ${LDC_EXE_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${EXTRA_CXXFLAGS}"
)
target_link_libraries(${LDC_EXE} ${LDC_LIB} ${LIBCONFIG++_LIBRARY})
if(MSVC)
    # Add a post build event in Visual Studio to copy the config file into Debug/Release folder
    add_custom_command(TARGET ${LDC_EXE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/bin/${LDC_EXE}.conf $<TARGET_FILE_DIR:${LDC_EXE}> COMMENT "Copy config file ${LDC_EXE}.conf")
endif()

# For use by the druntime/Phobos build system.
get_target_property(LDC_LOC ${LDC_EXE} LOCATION)

#
# Intrinsics module generation tools.
#

# The LLVM_INCLUDE_DIR definition is not always set, e.g. on Windows.
find_path(LLVM_INTRINSIC_TD_PATH "llvm/Intrinsics.td" PATHS ${LLVM_INCLUDE_DIRS} NO_DEFAULT_PATH)
if (${LLVM_INTRINSIC_TD_PATH} STREQUAL "LLVM_INTRINSIC_TD_PATH-NOTFOUND")
    message(SEND_ERROR "File llvm/Intrinsics.td not found")
else()
    message(STATUS "Using path for llvm/Intrinsics.td: ${LLVM_INTRINSIC_TD_PATH}")
endif()
add_definitions(-DLLVM_INTRINSIC_TD_PATH="${LLVM_INTRINSIC_TD_PATH}")

add_executable(gen_gccbuiltins utils/gen_gccbuiltins.cpp)

# Prior to LLVM 3.2, TableGen still uses RTTI, contrary to the rest of LLVM.
# Thus, we have to remove the -fno-rtti flag from the llvm-config output.
set(TABLEGEN_CXXFLAGS "${LLVM_CXXFLAGS}")
if(${LDC_LLVM_VER} LESS 302)
    string(REPLACE "-fno-rtti" "" TABLEGEN_CXXFLAGS ${TABLEGEN_CXXFLAGS})
endif()

set_target_properties(
    gen_gccbuiltins PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${TABLEGEN_CXXFLAGS}"
)
target_link_libraries(gen_gccbuiltins "${LLVM_LDFLAGS}" ${LLVM_LIBRARIES})
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(gen_gccbuiltins dl)
endif()

get_target_property(GEN_GCCBUILTINS_LOC gen_gccbuiltins LOCATION)

#
# LDMD
#
include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS(_SC_ARG_MAX "unistd.h" HAVE_SC_ARG_MAX)
if (HAVE_SC_ARG_MAX)
   add_definitions(-DHAVE_SC_ARG_MAX)
endif()

set_source_files_properties(dmd2/root/response.c dmd2/root/man.c PROPERTIES LANGUAGE CXX)
add_executable(${LDMD_EXE} dmd2/root/response.c dmd2/root/man.c driver/ldmd.cpp)
set_target_properties(${LDMD_EXE} PROPERTIES
    COMPILE_DEFINITIONS LDC_EXE_NAME="${LDC_EXE_NAME}"
    COMPILE_FLAGS "${LLVM_CXXFLAGS}"
    OUTPUT_NAME "${LDMD_EXE_NAME}"
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
# Same as above, LLVM_LDFLAGS should really be in LINK_FLAGS, but the LLVM libs
# use symbols from libdl, ..., so LLVM_LDFLAGS must come _after_ them in the
# command line. Maybe this could be improved using library groups, at least with
# GNU ld.
target_link_libraries(${LDMD_EXE} "${LLVM_LDFLAGS}" ${LLVM_LIBRARIES} "${LLVM_LDFLAGS}")

#
# Test and runtime targets. Note that enable_testing() is order-sensitive!
#
enable_testing()
add_subdirectory(runtime)
if(D_VERSION EQUAL 2)
    add_subdirectory(tests/d2)
endif()

#
# Install target.
#

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
endif()
