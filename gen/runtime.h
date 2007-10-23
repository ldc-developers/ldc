// D runtime support helpers

bool LLVM_D_InitRuntime();
void LLVM_D_FreeRuntime();

llvm::Function* LLVM_D_GetRuntimeFunction(llvm::Module* target, const char* name);

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name);
