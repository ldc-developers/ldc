#ifndef LLVMDC_GEN_TYPEINF_H
#define LLVMDC_GEN_TYPEINF_H

void DtoResolveTypeInfo(TypeInfoDeclaration* tid);
void DtoDeclareTypeInfo(TypeInfoDeclaration* tid);
void DtoConstInitTypeInfo(TypeInfoDeclaration* tid);
void DtoDefineTypeInfo(TypeInfoDeclaration* tid);

#endif
