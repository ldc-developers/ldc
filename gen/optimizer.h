#ifndef LDC_GEN_OPTIMIZER_H
#define LDC_GEN_OPTIMIZER_H

namespace llvm { class Module; }

bool ldc_optimize_module(llvm::Module* m);

// Determines whether the inliner will run in the -O<N> list of passes
bool doInline();
// Determines whether the inliner will be run at all.
bool willInline();

int optLevel();

bool optimize();

#endif

