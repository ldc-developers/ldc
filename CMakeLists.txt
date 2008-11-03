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
dmd/identifier.c
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
ir/irvar.cpp
)

set (LDC_GENERATED_SOURCE_FILES
${CMAKE_CURRENT_BINARY_DIR}/impcnvtab.c
${CMAKE_CURRENT_BINARY_DIR}/id.c
)
set (LDC_GENERATED_HEADER_FILES
${CMAKE_CURRENT_BINARY_DIR}/id.h
)

# some sources are generated
set_source_files_properties (
  ${LDC_GENERATED_SOURCE_FILES} 
  ${LDC_GENERATED_HEADER_FILES} 
  PROPERTIES GENERATED 1
)

# mark all source as c++
set_source_files_properties (
  ${LDC_SOURCE_FILES} ${LDC_GENERATED_SOURCE_FILES} 
  PROPERTIES LANGUAGE CXX
)

# build and run idgen and impcnvgen
set_source_files_properties (
  dmd/impcnvgen.c
  dmd/idgen.c
  PROPERTIES LANGUAGE CXX
)
add_executable (impcnvgen dmd/impcnvgen.c)
add_executable (idgen dmd/idgen.c)
set_target_properties (impcnvgen idgen PROPERTIES LINKER_LANGUAGE CXX)
get_target_property (IMPCNVGEN_EXE impcnvgen LOCATION)
get_target_property (IDGEN_EXE idgen LOCATION)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/impcnvtab.c
  COMMAND ${IMPCNVGEN_EXE}
  DEPENDS impcnvgen
)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/id.c ${CMAKE_CURRENT_BINARY_DIR}/id.h
  COMMAND ${IDGEN_EXE}
  DEPENDS idgen
)

# ldc itself
add_executable (ldc ${LDC_SOURCE_FILES} ${LDC_GENERATED_SOURCE_FILES} ${LDC_GENERATED_HEADER_FILES})


# setup DEFAULT_TARGET_TRIPLE
execute_process (COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/config.guess OUTPUT_VARIABLE DEFAULT_TARGET_TRIPLE_GUESS OUTPUT_STRIP_TRAILING_WHITESPACE)
set (DEFAULT_TARGET_TRIPLE ${DEFAULT_TARGET_TRIPLE_GUESS} CACHE STRING "the default target triple LDC should compile for")

# setup POSIX
if (WIN32)
  set (POSIX_GUESS 0)
else (WIN32)
  set (POSIX_GUESS 1)
endif (WIN32)
set (POSIX ${POSIX_GUESS} CACHE BOOL "is LDC build on POSIX?")

# setup defines
set (LDC_DEFINES
-DIN_LLVM
-D_DH
-DOPAQUE_VTBLS
-DUSE_BOEHM_GC=0
-DDMDV1
-DPOSIX=${POSIX}
-DDEFAULT_TARGET_TRIPLE=\\"${DEFAULT_TARGET_TRIPLE}\\"
-DX86_REVERSE_PARAMS=1
-DX86_PASS_IN_EAX=1)
execute_process (COMMAND llvm-config --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
# passing CXXFLAGS here is not good style
add_definitions (${LDC_DEFINES} ${LLVM_CXXFLAGS})

include_directories (. dmd ${CMAKE_CURRENT_BINARY_DIR})

# this is hackish, but works
target_link_libraries (ldc "`llvm-config --ldflags` `llvm-config --libs bitwriter linker ipo instrumentation backend`")

#TODO: runtime build
