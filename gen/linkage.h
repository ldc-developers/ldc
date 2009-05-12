#ifndef LDC_GEN_LINKAGE_H
#define LDC_GEN_LINKAGE_H

// Make it easier to test new linkage types

#  define TEMPLATE_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceODRLinkage
#  define TYPEINFO_LINKAGE_TYPE         llvm::GlobalValue::LinkOnceODRLinkage
// The One-Definition-Rule shouldn't matter for debug info, right?
#  define DEBUGINFO_LINKONCE_LINKAGE_TYPE \
                                        llvm::GlobalValue::LinkOnceAnyLinkage


#endif
