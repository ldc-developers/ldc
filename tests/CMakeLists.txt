set( LDC2_BIN          ${PROJECT_BINARY_DIR}/bin/${LDC_EXE} )
set( LDCPROFDATA_BIN   ${PROJECT_BINARY_DIR}/bin/ldc-profdata )
set( LDCPROFGEN_BIN    ${PROJECT_BINARY_DIR}/bin/ldc-profgen )
set( LDCPRUNECACHE_BIN ${PROJECT_BINARY_DIR}/bin/${LDCPRUNECACHE_EXE} )
set( LDCBUILDPLUGIN_BIN ${PROJECT_BINARY_DIR}/bin/${LDC_BUILD_PLUGIN_EXE} )
set( TIMETRACE2TXT_BIN ${PROJECT_BINARY_DIR}/bin/${TIMETRACE2TXT_EXE} )
set( LLVM_TOOLS_DIR    ${LLVM_ROOT_DIR}/bin )
set( LDC2_BIN_DIR      ${PROJECT_BINARY_DIR}/bin )
set( LDC2_LIB_DIR      ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX} )
set( TESTS_IR_DIR      ${CMAKE_CURRENT_SOURCE_DIR} )
set( PYTHON_EXE        python3)
if(WIN32)
    set(PYTHON_EXE python)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( DEFAULT_TARGET_BITS 64 )
else()
    set( DEFAULT_TARGET_BITS 32 )
endif()

configure_file(lit.site.cfg.in lit.site.cfg )
configure_file(runlit.py       runlit.py    COPYONLY)

add_test(NAME lit-tests
    COMMAND ${PYTHON_EXE} runlit.py -v .
)

