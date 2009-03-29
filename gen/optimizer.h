#ifndef LDC_GEN_OPTIMIZER_H
#define LDC_GEN_OPTIMIZER_H

namespace llvm { class Module; }

bool ldc_optimize_module(llvm::Module* m);

bool doInline();

int optLevel();

bool optimize();

#endif