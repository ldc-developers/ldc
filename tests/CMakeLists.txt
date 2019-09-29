set( LDC2_BIN          ${PROJECT_BINARY_DIR}/bin/${LDC_EXE} )
set( LDCPROFDATA_BIN   ${PROJECT_BINARY_DIR}/bin/ldc-profdata )
set( LDCPRUNECACHE_BIN ${PROJECT_BINARY_DIR}/bin/${LDCPRUNECACHE_EXE} )
set( LLVM_TOOLS_DIR    ${LLVM_ROOT_DIR}/bin )
set( LDC2_BIN_DIR      ${PROJECT_BINARY_DIR}/bin )
set( LDC2_LIB_DIR      ${PROJECT_BINARY_DIR}/lib${LIB_SUFFIX} )
set( TESTS_IR_DIR      ${CMAKE_CURRENT_SOURCE_DIR} )

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( DEFAULT_TARGET_BITS 64 )
else()
    set( DEFAULT_TARGET_BITS 32 )
endif()

function(get_host_arch var_name)
    # HACK!
    # LLVMConfig provide useful LLVM_NATIVE_ARCH variable which is not
    # provided from our FindLLVM nor from llvm-config.
    # Parsing LLVM_HOST_TARGET is not enough, because it may require
    # additional normalization (e.g. x86, x86_64, amd64, i686 -> x86), which
    # is already done in LLVM_NATIVE_ARCH
    include("${LLVM_CMAKEDIR}/LLVMConfig.cmake")
    include("${LLVM_CMAKEDIR}/LLVM-Config.cmake")
    set(${var_name} ${LLVM_NATIVE_ARCH} PARENT_SCOPE)
endfunction()

get_host_arch(LDC_HOST_ARCH)
message(STATUS "LDC_HOST_ARCH: ${LDC_HOST_ARCH}")

configure_file(lit.site.cfg.in lit.site.cfg )
configure_file(runlit.py       runlit.py    COPYONLY)

add_test(NAME lit-tests
    COMMAND python runlit.py -v .
)

