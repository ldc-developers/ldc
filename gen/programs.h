#ifndef LDC_GEN_PROGRAMS_H
#define LDC_GEN_PROGRAMS_H

#include "llvm/Support/Path.h"

llvm::sys::Path getGcc();
llvm::sys::Path getArchiver();

// For Windows with MS tool chain
llvm::sys::Path getLink();
llvm::sys::Path getLib();

#endif
