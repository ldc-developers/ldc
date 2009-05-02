#ifndef LDC_GEN_METADATA_H
#define LDC_GEN_METADATA_H

#include "gen/llvm-version.h"

#if LLVM_REV >= 68420
#  define USE_METADATA
#  define METADATA_LINKAGE_TYPE  llvm::GlobalValue::WeakODRLinkage
#endif

#endif
