#ifndef LDC_GEN_LLVMHELPERS_H
#define LDC_GEN_LLVMHELPERS_H

#include "gen/llvm.h"
#include "statement.h"

// this is used for tracking try-finally, synchronized and volatile scopes
struct EnclosingHandler
{
    virtual void emitCode(IRState* p) = 0;
};
struct EnclosingTryFinally : EnclosingHandler
{
    TryFinallyStatement* tf;
    void emitCode(IRState* p);
    EnclosingTryFinally(TryFinallyStatement* _tf) : tf(_tf) {}
};
struct EnclosingVolatile : EnclosingHandler
{
    VolatileStatement* v;
    void emitCode(IRState* p);
    EnclosingVolatile(VolatileStatement* _tf) : v(_tf) {}
};
struct EnclosingSynchro : EnclosingHandler
{
    SynchronizedStatement* s;
    void emitCode(IRState* p);
    EnclosingSynchro(SynchronizedStatement* _tf) : s(_tf) {}
};


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
void DtoAssert(Module* M, Loc loc, DValue* msg);

// return the LabelStatement from the current function with the given identifier or NULL if not found
LabelStatement* DtoLabelStatement(Identifier* ident);
// emit goto
void DtoGoto(Loc loc, Identifier* target);

// Generates IR for enclosing handlers between the current state and
// the scope created by the 'target' statement.
void DtoEnclosingHandlers(Loc loc, Statement* target);

/// Enters a critical section.
void DtoEnterCritical(LLValue* g);
/// leaves a critical section.
void DtoLeaveCritical(LLValue* g);

/// Enters a monitor lock.
void DtoEnterMonitor(LLValue* v);
/// Leaves a monitor lock.
void DtoLeaveMonitor(LLValue* v);

// nested variable and context helpers

/// Gets the context value for a call to a nested function or newing a nested
/// class with arbitrary nesting.
LLValue* DtoNestedContext(Loc loc, Dsymbol* sym);

/// Gets the DValue of a nested variable with arbitrary nesting.
DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd);

// basic operations
void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs);

/// Create a null DValue.
DValue* DtoNullValue(Type* t);

// casts
DValue* DtoCastInt(Loc& loc, DValue* val, Type* to);
DValue* DtoCastPtr(Loc& loc, DValue* val, Type* to);
DValue* DtoCastFloat(Loc& loc, DValue* val, Type* to);
DValue* DtoCastDelegate(Loc& loc, DValue* val, Type* to);
DValue* DtoCast(Loc& loc, DValue* val, Type* to);

// return the same val as passed in, modified to the target type, if possible, otherwise returns a new DValue
DValue* DtoPaintType(Loc& loc, DValue* val, Type* to);

// is template instance check, returns module where instantiated
TemplateInstance* DtoIsTemplateInstance(Dsymbol* s);

/// Generate code for the symbol.
/// Dispatches as appropriate.
void DtoResolveDsymbol(Dsymbol* dsym);

/// Generates the constant initializer for a global variable.
void DtoConstInitGlobal(VarDeclaration* vd);

// declaration inside a declarationexp
DValue* DtoDeclarationExp(Dsymbol* declaration);
LLValue* DtoRawVarDeclaration(VarDeclaration* var);

// initializer helpers
LLConstant* DtoConstInitializer(Loc loc, Type* type, Initializer* init);
LLConstant* DtoConstExpInit(Loc loc, Type* t, Expression* exp);
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

/// Fixup an overloaded intrinsic name string.
void DtoOverloadedIntrinsicName(TemplateInstance* ti, TemplateDeclaration* td, std::string& name);

/// Returns true if the symbol should be defined in the current module, not just declared.
bool mustDefineSymbol(Dsymbol* s);

/// Returns true if the symbol needs template linkage, or just external.
bool needsTemplateLinkage(Dsymbol* s);

/// Returns true if there is any unaligned type inside the aggregate.
bool hasUnalignedFields(Type* t);

///
DValue* DtoInlineAsmExpr(Loc loc, FuncDeclaration* fd, Expressions* arguments);

////////////////////////////////////////////
// gen/tocall.cpp stuff below
////////////////////////////////////////////

/// convert DMD calling conv to LLVM
unsigned DtoCallingConv(Loc loc, LINK l);

///
TypeFunction* DtoTypeFunction(DValue* fnval);

///
DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg);

///
LLValue* DtoCallableValue(DValue* fn);

///
const LLFunctionType* DtoExtractFunctionType(const LLType* type);

///
void DtoBuildDVarArgList(std::vector<LLValue*>& args, llvm::AttrListPtr& palist, TypeFunction* tf, Expressions* arguments, size_t argidx);

///
DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments);

#endif
