#if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${FILECHECK_SRC})
#    add_executable(FileCheck ${FILECHECK_SRC})
#    set_target_properties(
#        FileCheck PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
#        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
#        LINK_FLAGS "${SANITIZE_LDFLAGS}"
#    )
#    target_link_libraries(FileCheck  ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
#else()
#    message(STATUS "Skip building FileCheck (source not found), assuming it can be found in LLVM bin directory")
#endif()

# Build `not` for testing
#add_executable(not not.cpp)
#set_target_properties(
#    not PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
#    COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
#    LINK_FLAGS "${SANITIZE_LDFLAGS}"
#)
#target_link_libraries(not  ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
