project (ldc)

set (LDC_SOURCE_FILES 
dmd/access.c
dmd/array.c
dmd/arrayop.c
dmd/attrib.c
dmd/cast.c
dmd/class.c
dmd/clone.c
dmd/cond.c
dmd/constfold.c
dmd/dchar.c
dmd/declaration.c
dmd/delegatize.c
dmd/doc.c
dmd/dsymbol.c
dmd/dump.c
dmd/entity.c
dmd/enum.c
dmd/expression.c
dmd/func.c
dmd/gnuc.c
dmd/hdrgen.c
dmd/html.c
dmd/id.c
dmd/identifier.c
dmd/impcnvtab.c
dmd/import.c
dmd/inifile.c
dmd/init.c
dmd/inline.c
dmd/interpret.c
dmd/lexer.c
dmd/lstring.c
dmd/macro.c
dmd/man.c
dmd/mangle.c
dmd/mars.c
dmd/mem.c
dmd/module.c
dmd/mtype.c
dmd/opover.c
dmd/optimize.c
dmd/parse.c
dmd/root.c
dmd/scope.c
dmd/statement.c
dmd/staticassert.c
dmd/stringtable.c
dmd/struct.c
dmd/template.c
dmd/unialpha.c
dmd/utf.c
dmd/version.c
gen/aa.cpp
gen/arrays.cpp
gen/asmstmt.cpp
gen/binops.cpp
gen/classes.cpp
gen/complex.cpp
gen/dvalue.cpp
gen/dwarftypes.cpp
gen/functions.cpp
gen/irstate.cpp
gen/linker.cpp
gen/llvmhelpers.cpp
gen/logger.cpp
gen/optimizer.cpp
gen/runtime.cpp
gen/statements.cpp
gen/structs.cpp
gen/tocall.cpp
gen/tocsym.cpp
gen/todebug.cpp
gen/todt.cpp
gen/toir.cpp
gen/tollvm.cpp
gen/toobj.cpp
gen/typinf.cpp
ir/irfunction.cpp
ir/irlandingpad.cpp
ir/irmodule.cpp
ir/irstruct.cpp
ir/irsymbol.cpp
ir/irtype.cpp
ir/irvar.cpp)

add_executable (ldc ${LDC_SOURCE_FILES})
set_source_files_properties (${LDC_SOURCE_FILES} PROPERTIES LANGUAGE CXX)

execute_process (COMMAND llvm-config --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process (COMMAND llvm-config --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR)
#execute_process (COMMAND llvm-config --libs bitwriter linker ipo instrumentation backend OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/config.guess OUTPUT_VARIABLE DEFAULT_TRIPLE OUTPUT_STRIP_TRAILING_WHITESPACE)

set (LDC_DEFINES
-DIN_LLVM
-D_DH
-DOPAQUE_VTBLS
-DUSE_BOEHM_GC=0
-DDMDV1
-DPOSIX=1
-DDEFAULT_TARGET_TRIPLE=\\"${DEFAULT_TRIPLE}\\"
-DX86_REVERSE_PARAMS=1
-DX86_PASS_IN_EAX=1)

# passing CXXFLAGS here is not good style
add_definitions (${LDC_DEFINES} ${LLVM_CXXFLAGS})
include_directories (. dmd)

# this is hackish, but works
target_link_libraries (ldc "`llvm-config --ldflags` `llvm-config --libs bitwriter linker ipo instrumentation backend`")

#TODO: POSIX detection, runtime build
