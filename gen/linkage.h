#ifndef LDC_GEN_LINKAGE_H
#define LDC_GEN_LINKAGE_H

#include "gen/revisions.h"

// Make it easier to test new linkage types
// Also used to adapt to some changes in LLVM between 2.5 and 2.6


// LLVM r66339 introduces LinkOnceODRLinkage, which is just what we want here.
// (It also renamed LinkOnceLinkage, so this #if is needed for LDC to compile
// with both 2.5 and trunk)
#if LLVM_REV >= 66339
#  define TEMPLATE_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceODRLinkage
#  define TYPEINFO_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceODRLinkage
// The One-Definition-Rule shouldn't matter for debug info, right?
#  define DEBUGINFO_LINKONCE_LINKAGE_TYPE \
                                        llvm::GlobalValue::LinkOnceAnyLinkage

// For 2.5 and any LLVM revision before 66339 we want to use LinkOnceLinkage
// It's equivalent to LinkOnceAnyLinkage in trunk except that the inliner had a
// hack (removed in r66339) to allow inlining of templated functions even though
// LinkOnce doesn't technically allow that.
#else
#  define TEMPLATE_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceLinkage
#  define TYPEINFO_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceLinkage
#  define DEBUGINFO_LINKONCE_LINKAGE_TYPE \
                                        llvm::GlobalValue::LinkOnceLinkage
#endif

#endif
