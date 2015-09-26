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

#include "mtype.h"
#include "statement.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"
#include "ir/irfuncty.h"

// dynamic memory helpers
LLValue* DtoNew(Loc& loc, Type* newtype);
LLValue* DtoNewStruct(Loc& loc, TypeStruct* newtype);
void DtoDeleteMemory(Loc& loc, DValue* ptr);
void DtoDeleteStruct(Loc& loc, DValue* ptr);
void DtoDeleteClass(Loc& loc, DValue* inst);
void DtoDeleteInterface(Loc& loc, DValue* inst);
void DtoDeleteArray(Loc& loc, DValue* arr);

// emit an alloca
llvm::AllocaInst* DtoAlloca(Type* type, const char* name = "");
llvm::AllocaInst* DtoArrayAlloca(Type* type, unsigned arraysize, const char* name = "");
llvm::AllocaInst* DtoRawAlloca(LLType* lltype, size_t alignment, const char* name = "");
LLValue* DtoGcMalloc(Loc& loc, LLType* lltype, const char* name = "");

LLValue* DtoAllocaDump(DValue* val, const char* name = "");
LLValue* DtoAllocaDump(DValue* val, Type* asType, const char* name = "");
LLValue* DtoAllocaDump(DValue* val, LLType* asType, int alignment = 0, const char* name = "");
LLValue* DtoAllocaDump(LLValue* val, int alignment = 0, const char* name = "");
LLValue* DtoAllocaDump(LLValue* val, Type* asType, const char* name = "");
LLValue* DtoAllocaDump(LLValue* val, LLType* asType, int alignment = 0, const char* name = "");

// assertion generator
void DtoAssert(Module* M, Loc& loc, DValue* msg);

// returns module file name
LLValue* DtoModuleFileName(Module* M, const Loc& loc);

/// emits goto to LabelStatement with the target identifier
void DtoGoto(Loc &loc, LabelDsymbol *target);

/// Enters a critical section.
void DtoEnterCritical(Loc& loc, LLValue* g);
/// leaves a critical section.
void DtoLeaveCritical(Loc& loc, LLValue* g);

/// Enters a monitor lock.
void DtoEnterMonitor(Loc& loc, LLValue* v);
/// Leaves a monitor lock.
void DtoLeaveMonitor(Loc& loc, LLValue* v);

// basic operations
void DtoAssign(Loc& loc, DValue* lhs, DValue* rhs, int op = -1, bool canSkipPostblit = false);

DValue* DtoSymbolAddress(Loc& loc, Type* type, Declaration* decl);
llvm::Constant* DtoConstSymbolAddress(Loc& loc,Declaration* decl);

/// Create a null DValue.
DValue* DtoNullValue(Type* t, Loc loc = Loc());

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

/// Makes sure the declarations corresponding to the given D symbol have been
/// emitted to the currently processed LLVM module.
///
/// This means that dsym->ir can be expected to set to reasonable values.
///
/// This function does *not* emit any (function, variable) *definitions*; this
/// is done by Dsymbol::codegen.
void DtoResolveDsymbol(Dsymbol* dsym);
void DtoResolveVariable(VarDeclaration* var);

// declaration inside a declarationexp
void DtoVarDeclaration(VarDeclaration* var);
DValue* DtoDeclarationExp(Dsymbol* declaration);
LLValue* DtoRawVarDeclaration(VarDeclaration* var, LLValue* addr = 0);

// initializer helpers
LLConstant* DtoConstInitializer(Loc& loc, Type* type, Initializer* init);
LLConstant* DtoConstExpInit(Loc& loc, Type* targetType, Expression* exp);

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
LLValue* DtoBinNumericEquals(Loc& loc, DValue* lhs, DValue* rhs, TOK op);
LLValue* DtoBinFloatsEquals(Loc& loc, DValue* lhs, DValue* rhs, TOK op);

// target stuff
void findDefaultTarget();

/// Fixup an overloaded intrinsic name string.
void DtoOverloadedIntrinsicName(TemplateInstance* ti, TemplateDeclaration* td, std::string& name);

/// Returns true if there is any unaligned type inside the aggregate.
bool hasUnalignedFields(Type* t);

/// Returns a pointer to the given member field of an aggregate.
///
/// 'src' is a pointer to the start of the memory of an 'ad' instance.
LLValue* DtoIndexAggregate(LLValue* src, AggregateDeclaration* ad, VarDeclaration* vd);

/// Returns the index of a given member variable in the resulting LLVM type of
/// an aggregate.
///
/// This is only a valid operation if the field is known to be non-overlapping,
/// so that no byte-wise offset is needed.
unsigned getFieldGEPIndex(AggregateDeclaration* ad, VarDeclaration* vd);

///
DValue* DtoInlineAsmExpr(Loc& loc, FuncDeclaration* fd, Expressions* arguments);

/// Returns the size the LLVM type for a member variable of the given type will
/// take up in a struct (in bytes). This does not include padding in any way.
size_t getMemberSize(Type* type);

/// Returns the llvm::Value of the passed DValue, making sure that it is an
/// lvalue (has a memory address), so it can be passed to the D runtime
/// functions without problems.
LLValue* makeLValue(Loc& loc, DValue* value);

void callPostblit(Loc& loc, Expression *exp, LLValue *val);

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

///
IrFuncTy &DtoIrTypeFunction(DValue* fnval);
///
TypeFunction* DtoTypeFunction(DValue* fnval);

///
LLValue* DtoCallableValue(DValue* fn);

///
LLFunctionType* DtoExtractFunctionType(LLType* type);

/// Checks whether fndecl is an intrinsic that requires special lowering. If so,
/// emits the code for it and returns true, settings result to the resulting
/// DValue (if any). If the call does not correspond to a "magic" intrinsic,
/// i.e. should be turned into a normal function call, returns false.
bool DtoLowerMagicIntrinsic(IRState* p, FuncDeclaration* fndecl, CallExp *e, DValue*& result);

///
DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments, LLValue* retvar = 0);

Type* stripModifiers(Type* type, bool transitive = false);

void printLabelName(std::ostream& target, const char* func_mangle, const char* label_name);

void AppendFunctionToLLVMGlobalCtorsDtors(llvm::Function* func, const uint32_t priority, const bool isCtor);

template <typename T>
LLConstant* toConstantArray(LLType* ct, LLArrayType* at, T* str, size_t len, bool nullterm = true)
{
    std::vector<LLConstant*> vals;
    vals.reserve(len+1);
    for (size_t i = 0; i < len; ++i) {
        vals.push_back(LLConstantInt::get(ct, str[i], false));
    }
    if (nullterm)
        vals.push_back(LLConstantInt::get(ct, 0, false));
    return LLConstantArray::get(at, vals);
}


/// Tries to create an LLVM global with the given properties. If a variable with
/// the same mangled name already exists, checks if the types match and returns
/// it instead.
///
/// Necessary to support multiple declarations with the same mangled name, as
/// can be the case due to pragma(mangle).
llvm::GlobalVariable* getOrCreateGlobal(Loc& loc, llvm::Module& module,
    llvm::Type* type, bool isConstant, llvm::GlobalValue::LinkageTypes linkage,
    llvm::Constant* init, llvm::StringRef name, bool isThreadLocal = false);

FuncDeclaration* getParentFunc(Dsymbol* sym, bool stopOnStatic);

void Declaration_codegen(Dsymbol *decl);
void Declaration_codegen(Dsymbol *decl, IRState *irs);

DValue *toElem(Expression *e);
DValue *toElem(Expression *e, bool tryGetLvalue);
DValue *toElemDtor(Expression *e);
LLConstant *toConstElem(Expression *e, IRState *p);

#if LDC_LLVM_VER >= 307
bool supportsCOMDAT();

#define SET_COMDAT(x,m) if (supportsCOMDAT()) x->setComdat(m.getOrInsertComdat(x->getName()))

#else

#define supportsCOMDAT() false
#define SET_COMDAT(x,m)

#endif

#endif
