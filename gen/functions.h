#ifndef LLVMDC_GEN_FUNCTIONS_H
#define LLVMDC_GEN_FUNCTIONS_H

const llvm::FunctionType* DtoFunctionType(Type* t, const llvm::Type* thistype, bool ismain = false);
const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl);

void DtoResolveFunction(FuncDeclaration* fdecl);
void DtoDeclareFunction(FuncDeclaration* fdecl);
void DtoDefineFunc(FuncDeclaration* fd);

void DtoMain();

#endif
