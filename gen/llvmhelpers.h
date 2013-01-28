//===-- gen/llvmhelpers.h - General LLVM codegen helpers --------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// General codegen helper constructs.
//
// TODO: Merge with gen/tollvm.h, then refactor into sensible parts.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LLVMHELPERS_H
#define LDC_GEN_LLVMHELPERS_H

#include "gen/llvm.h"
#include "gen/dvalue.h"

#include "statement.h"
#include "mtype.h"

// this is used for tracking try-finally, synchronized and volatile scopes
struct EnclosingHandler
{
    virtual void emitCode(IRState* p) = 0;
};
struct EnclosingTryFinally : EnclosingHandler
{
    TryFinallyStatement* tf;
    llvm::BasicBlock* landingPad;
    void emitCode(IRState* p);
    EnclosingTryFinally(TryFinallyStatement* _tf, llvm::BasicBlock* _pad)
    : tf(_tf), landingPad(_pad) {}
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
llvm::AllocaInst* DtoAlloca(Type* type, const char* name = "");
llvm::AllocaInst* DtoArrayAlloca(Type* type, unsigned arraysize, const char* name = "");
llvm::AllocaInst* DtoRawAlloca(LLType* lltype, size_t alignment, const char* name = "");
LLValue* DtoGcMalloc(LLType* lltype, const char* name = "");

// assertion generator
void DtoAssert(Module* M, Loc loc, DValue* msg);

// return the LabelStatement from the current function with the given identifier or NULL if not found
LabelStatement* DtoLabelStatement(Identifier* ident);

/// emits goto to LabelStatement with the target identifier
/// the sourceFinally is only used for error checking
void DtoGoto(Loc loc, Identifier* target, TryFinallyStatement* sourceFinally);

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

// basic operations
void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs, int op = -1, bool canSkipPostblit = false);

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
TemplateInstance* DtoIsTemplateInstance(Dsymbol* s, bool checkLiteralOwner = false);

/// Generate code for the symbol.
/// Dispatches as appropriate.
void DtoResolveDsymbol(Dsymbol* dsym);

/// Generates the constant initializer for a global variable.
void DtoConstInitGlobal(VarDeclaration* vd);

// declaration inside a declarationexp
void DtoVarDeclaration(VarDeclaration* var);
DValue* DtoDeclarationExp(Dsymbol* declaration);
LLValue* DtoRawVarDeclaration(VarDeclaration* var, LLValue* addr = 0);

// initializer helpers
LLType* DtoConstInitializerType(Type* type, Initializer* init);
LLConstant* DtoConstInitializer(Loc loc, Type* type, Initializer* init);
LLConstant* DtoConstExpInit(Loc loc, Type* t, Expression* exp);
DValue* DtoInitializer(LLValue* target, Initializer* init);

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
LLValue* DtoBinNumericEquals(Loc loc, DValue* lhs, DValue* rhs, TOK op);
LLValue* DtoBinFloatsEquals(Loc loc, DValue* lhs, DValue* rhs, TOK op);

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

/// Create the IrModule if necessary and returns it.
IrModule* getIrModule(Module* M);

/// Update an offset to make sure it follows both the D and LLVM alignments.
/// Returns the offset rounded up to the closest safely aligned offset.
size_t realignOffset(size_t offset, Type* type);

/// Returns the llvm::Value of the passed DValue, making sure that it is an
/// lvalue (has a memory address), so it can be passed to the D runtime
/// functions without problems.
LLValue* makeLValue(Loc& loc, DValue* value);

#if DMDV2
void callPostblit(Loc &loc, Expression *exp, LLValue *val);
#endif

/// Returns whether the given variable is a DMD-internal "ref variable".
///
/// D doesn't have reference variables (the ref keyword is only usable in
/// function signatures and foreach headers), but the DMD frontend internally
/// creates them in cases like lowering a ref foreach to a for loop or the
/// implicit __result variable for ref-return functions with out contracts.
bool isSpecialRefVar(VarDeclaration* vd);

/// Returns whether the type is unsigned in LLVM terms, which also includes
/// pointers.
bool isLLVMUnsigned(Type* t);

/// Converts a DMD comparison operation token into the corresponding LLVM icmp
/// predicate for the given operand signedness.
///
/// For some operations, the result can be a constant. In this case outConst is
/// set to it, otherwise outPred is set to the predicate to use.
void tokToIcmpPred(TOK op, bool isUnsigned, llvm::ICmpInst::Predicate* outPred,
    llvm::Value** outConst);

////////////////////////////////////////////
// gen/tocall.cpp stuff below
////////////////////////////////////////////

/// convert DMD calling conv to LLVM
llvm::CallingConv::ID DtoCallingConv(Loc loc, LINK l);

///
TypeFunction* DtoTypeFunction(DValue* fnval);

///
DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg);

///
LLValue* DtoCallableValue(DValue* fn);

///
LLFunctionType* DtoExtractFunctionType(LLType* type);

///
#if LDC_LLVM_VER >= 303
void DtoBuildDVarArgList(std::vector<LLValue*>& args, llvm::AttributeSet& palist, TypeFunction* tf, Expressions* arguments, size_t argidx);
#else
void DtoBuildDVarArgList(std::vector<LLValue*>& args, llvm::AttrListPtr& palist, TypeFunction* tf, Expressions* arguments, size_t argidx);
#endif

///
DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments);

Type* stripModifiers(Type* type);

void printLabelName(std::ostream& target, const char* func_mangle, const char* label_name);

void AppendFunctionToLLVMGlobalCtorsDtors(llvm::Function* func, const uint32_t priority, const bool isCtor);

#endif
