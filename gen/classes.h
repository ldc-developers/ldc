#ifndef LLVMDC_GEN_CLASSES_H
#define LLVMDC_GEN_CLASSES_H

/**
 * Resolves the llvm type for a class declaration
 */
void DtoResolveClass(ClassDeclaration* cd);

/**
 * Provides the llvm declaration for a class declaration
 */
void DtoDeclareClass(ClassDeclaration* cd);

/**
 * Constructs the constant initializer for a class declaration
 */
void DtoConstInitClass(ClassDeclaration* cd);

/**
 * Provides the llvm definition for a class declaration
 */
void DtoDefineClass(ClassDeclaration* cd);

void DtoDeclareClassInfo(ClassDeclaration* cd);
void DtoDefineClassInfo(ClassDeclaration* cd);

DValue* DtoNewClass(TypeClass* type, NewExp* newexp);
void DtoInitClass(TypeClass* tc, llvm::Value* dst);
DValue* DtoCallClassCtor(TypeClass* type, CtorDeclaration* ctor, Array* arguments, llvm::Value* mem);
void DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void DtoFinalizeClass(llvm::Value* inst);

DValue* DtoCastClass(DValue* val, Type* to);
DValue* DtoDynamicCastObject(DValue* val, Type* to);

DValue* DtoCastInterfaceToObject(DValue* val, Type* to);
DValue* DtoDynamicCastInterface(DValue* val, Type* to);

llvm::Value* DtoIndexClass(llvm::Value* ptr, ClassDeclaration* cd, Type* t, unsigned os, std::vector<unsigned>& idxs);

llvm::Value* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl);

#endif
