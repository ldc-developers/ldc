# "tools" are supposed to be packaged with LDC.
# (in contrast to "utils" which are only used for building / testing)

option(LDC_BUNDLE_LLVM_TOOLS "Build and install ldc-profgen and ldc-profdata utilities instead of using their llvm counterparts" TRUE)

#############################################################################
# Build ldc-prune-cache
set(LDCPRUNECACHE_EXE ldc-prune-cache)
set(LDCPRUNECACHE_EXE ${LDCPRUNECACHE_EXE} PARENT_SCOPE) # needed for correctly populating lit.site.cfg.in
set(LDCPRUNECACHE_EXE_NAME ${PROGRAM_PREFIX}${LDCPRUNECACHE_EXE}${PROGRAM_SUFFIX})
set(LDCPRUNECACHE_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDCPRUNECACHE_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(LDCPRUNECACHE_D_SRC
    ${PROJECT_SOURCE_DIR}/tools/ldc-prune-cache.d
    ${PROJECT_SOURCE_DIR}/driver/cache_pruning.d
)
build_d_executable(
    "${LDCPRUNECACHE_EXE}"
    "${LDCPRUNECACHE_EXE_FULL}"
    "${LDCPRUNECACHE_D_SRC}"
    "${DFLAGS_BUILD_TYPE}"
    "${FULLY_STATIC_LDFLAG}"
    ""
    ""
    ${COMPILE_D_MODULES_SEPARATELY}
)
install(PROGRAMS ${LDCPRUNECACHE_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

if(LDC_BUNDLE_LLVM_TOOLS)
  #############################################################################
  # Build ldc-profdata for converting profile data formats (source version depends on LLVM version)
  set(LDCPROFDATA_SRC ldc-profdata/llvm-profdata-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.cpp)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LDCPROFDATA_SRC})
      find_package(Threads)
      add_executable(ldc-profdata ${LDCPROFDATA_SRC})
      set_target_properties(
        ldc-profdata PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
        COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
        LINK_FLAGS "${SANITIZE_LDFLAGS} ${FULLY_STATIC_LDFLAG}"
      )
      set(link_llvmdebuginfod)
      if(LLVM_VERSION_MAJOR GREATER 19)
        if(MSVC)
          set(link_llvmdebuginfod "${LLVM_LIBRARY_DIRS}/LLVMDebuginfod.lib")
        else()
          set(link_llvmdebuginfod -lLLVMDebuginfod)
        endif()
      endif()
      target_link_libraries(ldc-profdata ${link_llvmdebuginfod} ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS} ${CMAKE_THREAD_LIBS_INIT})
      install(TARGETS ldc-profdata DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

      # Set path to executable, used by the lit testsuite.
      set(LDCPROFDATA_BIN ${PROJECT_BINARY_DIR}/bin/ldc-profdata PARENT_SCOPE)
  else()
      message(WARNING "ldc-profdata source (${LDCPROFDATA_SRC}) not found")
  endif()

  #############################################################################
  # Build ldc-profgen utility that generates a profile data file from given perf script
  # data files for sample-based profile guided optimization (-fprofile-sample-use).
  # https://llvm.org/docs/CommandGuide/llvm-profgen.html
  # The source in ldc-profgen/ldc-profgen-xx.x is an unmodified copy of llvm's llvm-profgen source dir.
  macro(add_llvm_tool llvm_name)
    string(REPLACE "llvm-" "ldc-" ldc_name ${llvm_name})
    message(STATUS "Configuring ${ldc_name} build target")
    add_executable(${ldc_name} ${ARGN})
    set_target_properties(
      ${ldc_name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
      COMPILE_FLAGS "${LLVM_CXXFLAGS} ${LDC_CXXFLAGS}"
      LINK_FLAGS "${SANITIZE_LDFLAGS} ${FULLY_STATIC_LDFLAG}"
    )
    target_link_libraries(${ldc_name} ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LDFLAGS})
    install(TARGETS ${ldc_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endmacro()
  if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ldc-profgen/ldc-profgen-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR})
    add_subdirectory(ldc-profgen/ldc-profgen-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR})
    # Set path to executable, potentially to be used by the lit tests in the future
    set(LDCPROFGEN_BIN ${PROJECT_BINARY_DIR}/bin/ldc-profgen PARENT_SCOPE)
  else()
    message(WARNING "ldc-profgen source not found (${CMAKE_CURRENT_SOURCE_DIR}/ldc-profgen/ldc-profgen-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR})")
  endif()
else()
  message(STATUS "Using llvm provided llvm-profdata and llvm-profgen")
  set(LDCPROFDATA_BIN ${LLVM_ROOT_DIR}/bin/llvm-profdata PARENT_SCOPE)
  set(LDCPROFGEN_BIN ${LLVM_ROOT_DIR}/bin/llvm-profgen PARENT_SCOPE)
endif()

#############################################################################
# Build timetrace2txt
set(TIMETRACE2TXT_EXE timetrace2txt)
set(TIMETRACE2TXT_EXE ${TIMETRACE2TXT_EXE} PARENT_SCOPE) # needed for correctly populating lit.site.cfg.in
set(TIMETRACE2TXT_EXE_NAME ${PROGRAM_PREFIX}${TIMETRACE2TXT_EXE}${PROGRAM_SUFFIX})
set(TIMETRACE2TXT_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${TIMETRACE2TXT_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
set(TIMETRACE2TXT_D_SRC
    ${PROJECT_SOURCE_DIR}/tools/timetrace2txt.d
)
build_d_executable(
    "${TIMETRACE2TXT_EXE}"
    "${TIMETRACE2TXT_EXE_FULL}"
    "${TIMETRACE2TXT_D_SRC}"
    "${DFLAGS_BUILD_TYPE}"
    "${FULLY_STATIC_LDFLAG}"
    ""
    ""
    ${COMPILE_D_MODULES_SEPARATELY}
)
install(PROGRAMS ${TIMETRACE2TXT_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#############################################################################
# Only build ldc-build-plugin tool for platforms where plugins are actually enabled.
if(LDC_ENABLE_PLUGINS)
    configure_file(${PROJECT_SOURCE_DIR}/tools/ldc-build-plugin.d.in ${PROJECT_BINARY_DIR}/ldc-build-plugin.d @ONLY)
    set(LDC_BUILD_PLUGIN_EXE ldc-build-plugin)
    set(LDC_BUILD_PLUGIN_EXE ${LDC_BUILD_PLUGIN_EXE} PARENT_SCOPE) # needed for correctly populating lit.site.cfg.in
    set(LDC_BUILD_PLUGIN_EXE_NAME ${PROGRAM_PREFIX}${LDC_BUILD_PLUGIN_EXE}${PROGRAM_SUFFIX})
    set(LDC_BUILD_PLUGIN_EXE_FULL ${PROJECT_BINARY_DIR}/bin/${LDC_BUILD_PLUGIN_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX})
    build_d_executable(
        "${LDC_BUILD_PLUGIN_EXE}"
        "${LDC_BUILD_PLUGIN_EXE_FULL}"
        "${PROJECT_BINARY_DIR}/ldc-build-plugin.d"
        "${DFLAGS_BUILD_TYPE}"
        "" # must not be fully static for std.net.curl support (loading libcurl dynamically)
        "${PROJECT_SOURCE_DIR}/tools/ldc-build-plugin.d.in"
        ""
        ${COMPILE_D_MODULES_SEPARATELY}
    )
    install(PROGRAMS ${LDC_BUILD_PLUGIN_EXE_FULL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

