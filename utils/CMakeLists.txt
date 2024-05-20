# Build gen_gccbuiltins tools to generate D module from LLVM's list of
# GCC-style intrinsics.

if(CMAKE_CROSSCOMPILING)
    message(STATUS "NOT building gen_gccbuiltins utility when cross-compiling (you can use host LDC's ldc/gccbuiltins_*.di files)")
    return()
endif()

# The LLVM_INCLUDE_DIR definition is not always set, e.g. on Windows.
# strip off anything but the first path
string(REGEX REPLACE ";.*$" "" LLVM_INC_DIR "${LLVM_INCLUDE_DIRS}")
find_path(LLVM_INTRINSIC_TD_PATH "Intrinsics.td" PATHS ${LLVM_INC_DIR}/llvm ${LLVM_INC_DIR}/llvm/IR NO_DEFAULT_PATH)
if (${LLVM_INTRINSIC_TD_PATH} STREQUAL "LLVM_INTRINSIC_TD_PATH-NOTFOUND")
    message(SEND_ERROR "File Intrinsics.td not found")
else()
    string(REGEX REPLACE "/llvm(/IR)?$" "" LLVM_INTRINSIC_TD_PATH ${LLVM_INTRINSIC_TD_PATH})
    message(STATUS "Using path for Intrinsics.td: ${LLVM_INTRINSIC_TD_PATH}")
endif()

add_executable(gen_gccbuiltins gen_gccbuiltins.cpp)

set_target_properties(
    gen_gccbuiltins PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} \"-DLLVM_INTRINSIC_TD_PATH=R\\\"(${LLVM_INTRINSIC_TD_PATH})\\\"\""
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(gen_gccbuiltins ${LLVM_TABLEGEN_LIBRARY} ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(gen_gccbuiltins dl)
endif()

if ((TARGET FileCheck) OR (EXISTS ${LLVM_ROOT_DIR}/bin/FileCheck))
  # already provided by LLVM
  message(STATUS "Skip building FileCheck, it is already provided by LLVM")
else()
# Build FileCheck for testing (build source version depending on LLVM version)
set(FILECHECK_SRC FileCheck-${LLVM_VERSION_MAJOR}.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${FILECHECK_SRC})
    add_executable(FileCheck ${FILECHECK_SRC})
    set_target_properties(
        FileCheck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS}"
    )
    target_link_libraries(FileCheck LLVMFileCheck)
    target_link_libraries(FileCheck ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
    message(STATUS "Building FileCheck from LDC source tree")
else()
    message(STATUS "Skip building FileCheck (source not found), assuming it can be found in LLVM bin directory")
endif()
endif()

if ((TARGET not) OR (EXISTS ${LLVM_ROOT_DIR}/bin/not))
  # already provided by LLVM
else()
# Build `not` for testing
add_executable(not not.cpp)
set_target_properties(
    not PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
    LINK_FLAGS "${SANITIZE_LDFLAGS}"
)
target_link_libraries(not  ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
endif()

if ((TARGET split-file) OR (EXISTS ${LLVM_ROOT_DIR}/bin/split-file))
  # already provided by LLVM
  message(STATUS "Skip building split-file, it is already provided by LLVM")
else()
# Build split-file for testing
set(SPLITFILE_SRC split-file.cpp)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SPLITFILE_SRC})
    add_executable(split-file ${SPLITFILE_SRC})
    set_target_properties(
        split-file PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS}"
    )
    target_link_libraries(split-file ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
    message(STATUS "Building split-file from LDC source tree")
else()
    message(STATUS "Skip building split-file (source not found), assuming it can be found in LLVM bin directory")
endif()
endif()
