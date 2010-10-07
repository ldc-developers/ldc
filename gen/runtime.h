#ifndef LDC_GEN_RUNTIME_H_
#define LDC_GEN_RUNTIME_H_

// D runtime support helpers

bool LLVM_D_InitRuntime();
void LLVM_D_FreeRuntime();

llvm::Function* LLVM_D_GetRuntimeFunction(llvm::Module* target, const char* name);

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name);

#if DMDV1
#define _d_allocclass "_d_allocclass"
#define _adEq "_adEq"
#else
#define _d_allocclass "_d_newclass"
#define _adEq "_adEq2"
#endif

#endif // LDC_GEN_RUNTIME_H_
