#ifndef LLVMDC_GEN_FUNCTIONS_H
#define LLVMDC_GEN_FUNCTIONS_H

const llvm::FunctionType* DtoFunctionType(Type* t, const LLType* thistype, bool ismain = false);
const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl);

const llvm::FunctionType* DtoBaseFunctionType(FuncDeclaration* fdecl);

void DtoResolveFunction(FuncDeclaration* fdecl);
void DtoDeclareFunction(FuncDeclaration* fdecl);
void DtoDefineFunc(FuncDeclaration* fd);

DValue* DtoArgument(Argument* fnarg, Expression* argexp);
void DtoVariadicArgument(Expression* argexp, LLValue* dst);

void DtoMain();

#endif
