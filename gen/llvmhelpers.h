#ifndef LLVMDC_GEN_LLVMHELPERS_H
#define LLVMDC_GEN_LLVMHELPERS_H

#include "statement.h"

// dynamic memory helpers
LLValue* DtoNew(Type* newtype);
void DtoDeleteMemory(LLValue* ptr);
void DtoDeleteClass(LLValue* inst);
void DtoDeleteInterface(LLValue* inst);
void DtoDeleteArray(DValue* arr);

// assertion generator
void DtoAssert(Loc* loc, DValue* msg);

// return the LabelStatement from the current function with the given identifier or NULL if not found
LabelStatement* DtoLabelStatement(Identifier* ident);
// emit goto
void DtoGoto(Loc* loc, Identifier* target, TryFinallyStatement* enclosingtryfinally);

// generates IR for finally blocks between the 'start' and 'end' statements
// will begin with the finally block belonging to 'start' and does not include
// the finally block of 'end'
void DtoFinallyBlocks(TryFinallyStatement* start, TryFinallyStatement* end);

// nested variable/class helpers
LLValue* DtoNestedContext(FuncDeclaration* func);
LLValue* DtoNestedVariable(VarDeclaration* vd);

// basic operations
void DtoAssign(DValue* lhs, DValue* rhs);

// casts
DValue* DtoCastInt(DValue* val, Type* to);
DValue* DtoCastPtr(DValue* val, Type* to);
DValue* DtoCastFloat(DValue* val, Type* to);
DValue* DtoCast(DValue* val, Type* to);

// is template instance check
bool DtoIsTemplateInstance(Dsymbol* s);

// generates lazy static initialization code for a global variable
void DtoLazyStaticInit(bool istempl, LLValue* gvar, Initializer* init, Type* t);

// these are all basically drivers for the codegeneration called by the main loop
void DtoResolveDsymbol(Dsymbol* dsym);
void DtoDeclareDsymbol(Dsymbol* dsym);
void DtoDefineDsymbol(Dsymbol* dsym);
void DtoConstInitDsymbol(Dsymbol* dsym);
void DtoConstInitGlobal(VarDeclaration* vd);
void DtoEmptyResolveList();
void DtoEmptyDeclareList();
void DtoEmptyConstInitList();
void DtoEmptyAllLists();
void DtoForceDeclareDsymbol(Dsymbol* dsym);
void DtoForceConstInitDsymbol(Dsymbol* dsym);
void DtoForceDefineDsymbol(Dsymbol* dsym);

// initializer helpers
LLConstant* DtoConstInitializer(Type* type, Initializer* init);
LLConstant* DtoConstFieldInitializer(Type* type, Initializer* init);
DValue* DtoInitializer(Initializer* init);

// annotation generator
void DtoAnnotation(const char* str);

// getting typeinfo of type, base=true casts to object.TypeInfo
LLConstant* DtoTypeInfoOf(Type* ty, bool base=true);

// binary operations
DValue* DtoBinAdd(DValue* lhs, DValue* rhs);
DValue* DtoBinSub(DValue* lhs, DValue* rhs);
DValue* DtoBinMul(DValue* lhs, DValue* rhs);
DValue* DtoBinDiv(DValue* lhs, DValue* rhs);
DValue* DtoBinRem(DValue* lhs, DValue* rhs);

// target stuff
void findDefaultTarget();

#endif
