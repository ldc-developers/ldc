#ifndef LDC_GEN_CLASSES_H
#define LDC_GEN_CLASSES_H

#include "gen/structs.h"

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

DValue* DtoNewClass(Loc loc, TypeClass* type, NewExp* newexp);
void DtoInitClass(TypeClass* tc, LLValue* dst);
DValue* DtoCallClassCtor(TypeClass* type, CtorDeclaration* ctor, Array* arguments, LLValue* mem);
void DtoFinalizeClass(LLValue* inst);

DValue* DtoCastClass(DValue* val, Type* to);
DValue* DtoDynamicCastObject(DValue* val, Type* to);

DValue* DtoCastInterfaceToObject(DValue* val, Type* to);
DValue* DtoDynamicCastInterface(DValue* val, Type* to);

LLValue* DtoIndexClass(LLValue* src, ClassDeclaration* sd, VarDeclaration* vd);

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl);

#endif
