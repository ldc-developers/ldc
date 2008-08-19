#ifndef LLVMDC_GEN_LLVMHELPERS_H
#define LLVMDC_GEN_LLVMHELPERS_H

#include "gen/llvm.h"
#include "statement.h"

// dynamic memory helpers
LLValue* DtoNew(Type* newtype);
void DtoDeleteMemory(LLValue* ptr);
void DtoDeleteClass(LLValue* inst);
void DtoDeleteInterface(LLValue* inst);
void DtoDeleteArray(DValue* arr);

// emit an alloca
llvm::AllocaInst* DtoAlloca(const LLType* lltype, const std::string& name = "");
llvm::AllocaInst* DtoAlloca(const LLType* lltype, LLValue* arraysize, const std::string& name = "");

// assertion generator
void DtoAssert(Loc* loc, DValue* msg);

// return the LabelStatement from the current function with the given identifier or NULL if not found
LabelStatement* DtoLabelStatement(Identifier* ident);
// emit goto
void DtoGoto(Loc* loc, Identifier* target, EnclosingHandler* enclosingtryfinally, TryFinallyStatement* sourcetf);

// generates IR for finally blocks between the 'start' and 'end' statements
// will begin with the finally block belonging to 'start' and does not include
// the finally block of 'end'
void DtoEnclosingHandlers(EnclosingHandler* start, EnclosingHandler* end);

// enters a critical section
void DtoEnterCritical(LLValue* g);
// leaves a critical section
void DtoLeaveCritical(LLValue* g);

// enters a monitor lock
void DtoEnterMonitor(LLValue* v);
// leaves a monitor lock
void DtoLeaveMonitor(LLValue* v);

// nested variable and context helpers

// gets the context value for a call to a nested function or newing a class, with arbitrary nesting
LLValue* DtoNestedContext(Loc loc, Dsymbol* sym);
// gets the dvalue of a nested variable with arbitrary nesting
DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd);

// basic operations
void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs);

// create a null dvalue
DValue* DtoNullValue(Type* t);

// casts
DValue* DtoCastInt(Loc& loc, DValue* val, Type* to);
DValue* DtoCastPtr(Loc& loc, DValue* val, Type* to);
DValue* DtoCastFloat(Loc& loc, DValue* val, Type* to);
DValue* DtoCastDelegate(Loc& loc, DValue* val, Type* to);
DValue* DtoCast(Loc& loc, DValue* val, Type* to);

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

// declaration inside a declarationexp
DValue* DtoDeclarationExp(Dsymbol* declaration);

// initializer helpers
LLConstant* DtoConstInitializer(Type* type, Initializer* init);
LLConstant* DtoConstFieldInitializer(Type* type, Initializer* init);
DValue* DtoInitializer(LLValue* target, Initializer* init);

// annotation generator
void DtoAnnotation(const char* str);

// getting typeinfo of type, base=true casts to object.TypeInfo
LLConstant* DtoTypeInfoOf(Type* ty, bool base=true);

// binary operations
DValue* DtoBinAdd(DValue* lhs, DValue* rhs);
DValue* DtoBinSub(DValue* lhs, DValue* rhs);
// these binops need an explicit result type to handling
// to give 'ifloat op float' and 'float op ifloat' the correct type
DValue* DtoBinMul(Type* resulttype, DValue* lhs, DValue* rhs);
DValue* DtoBinDiv(Type* resulttype, DValue* lhs, DValue* rhs);
DValue* DtoBinRem(Type* resulttype, DValue* lhs, DValue* rhs);

// target stuff
void findDefaultTarget();

/// Converts any value to a boolean (llvm i1)
LLValue* DtoBoolean(Loc& loc, DValue* dval);

/// get the default initializer of the type
LLConstant* DtoDefaultInit(Type* t);

// fixup an overloaded intrinsic name string
void DtoOverloadedIntrinsicName(TemplateInstance* ti, TemplateDeclaration* td, std::string& name);

////////////////////////////////////////////
// gen/tocall.cpp stuff below
////////////////////////////////////////////

/// convert DMD calling conv to LLVM
unsigned DtoCallingConv(LINK l);

///
TypeFunction* DtoTypeFunction(DValue* fnval);

///
DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg);

///
LLValue* DtoCallableValue(DValue* fn);

///
const LLFunctionType* DtoExtractFunctionType(const LLType* type);

///
void DtoBuildDVarArgList(std::vector<LLValue*>& args, llvm::PAListPtr& palist, TypeFunction* tf, Expressions* arguments, size_t argidx);

///
DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments);

#endif
