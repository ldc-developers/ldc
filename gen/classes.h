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

void DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void DtoInitClass(TypeClass* tc, llvm::Value* dst);

DValue* DtoCastClass(DValue* val, Type* to);
DValue* DtoDynamicCastObject(DValue* val, Type* to);
DValue* DtoCastInterfaceToObject(DValue* val, Type* to);

#endif
