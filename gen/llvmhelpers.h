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

#pragma once

#include "dmd/mtype.h"
#include "dmd/statement.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"
#include "ir/irfuncty.h"

struct IRState;

// An arrayreference type with initializer_list support (C++11):
template <class T> using ArrayParam = llvm::ArrayRef<T>;

llvm::LLVMContext& getGlobalContext();

// dynamic memory helpers
LLValue *DtoNew(Loc &loc, Type *newtype);
LLValue *DtoNewStruct(Loc &loc, TypeStruct *newtype);
void DtoDeleteMemory(Loc &loc, DValue *ptr);
void DtoDeleteStruct(Loc &loc, DValue *ptr);
void DtoDeleteClass(Loc &loc, DValue *inst);
void DtoDeleteInterface(Loc &loc, DValue *inst);
void DtoDeleteArray(Loc &loc, DValue *arr);

unsigned DtoAlignment(Type *type);
unsigned DtoAlignment(VarDeclaration *vd);

// emit an alloca
llvm::AllocaInst *DtoAlloca(Type *type, const char *name = "");
llvm::AllocaInst *DtoAlloca(VarDeclaration *vd, const char *name = "");
llvm::AllocaInst *DtoArrayAlloca(Type *type, unsigned arraysize,
                                 const char *name = "");
llvm::AllocaInst *DtoRawAlloca(LLType *lltype, size_t alignment,
                               const char *name = "");
LLValue *DtoGcMalloc(Loc &loc, LLType *lltype, const char *name = "");

LLValue *DtoAllocaDump(DValue *val, const char *name = "");
LLValue *DtoAllocaDump(DValue *val, int alignment, const char *name = "");
LLValue *DtoAllocaDump(DValue *val, Type *asType, const char *name = "");
LLValue *DtoAllocaDump(DValue *val, LLType *asType, int alignment = 0,
                       const char *name = "");
LLValue *DtoAllocaDump(LLValue *val, int alignment = 0, const char *name = "");
LLValue *DtoAllocaDump(LLValue *val, Type *asType, const char *name = "");
LLValue *DtoAllocaDump(LLValue *val, LLType *asType, int alignment = 0,
                       const char *name = "");

// assertion generator
void DtoAssert(Module *M, Loc &loc, DValue *msg);
void DtoCAssert(Module *M, Loc &loc, LLValue *msg);

// returns module file name
LLConstant *DtoModuleFileName(Module *M, const Loc &loc);

/// emits goto to LabelStatement with the target identifier
void DtoGoto(Loc &loc, LabelDsymbol *target);

/// Enters a critical section.
void DtoEnterCritical(Loc &loc, LLValue *g);
/// leaves a critical section.
void DtoLeaveCritical(Loc &loc, LLValue *g);

/// Enters a monitor lock.
void DtoEnterMonitor(Loc &loc, LLValue *v);
/// Leaves a monitor lock.
void DtoLeaveMonitor(Loc &loc, LLValue *v);

// basic operations
void DtoAssign(Loc &loc, DValue *lhs, DValue *rhs, int op,
               bool canSkipPostblit = false);

DValue *DtoSymbolAddress(Loc &loc, Type *type, Declaration *decl);
llvm::Constant *DtoConstSymbolAddress(Loc &loc, Declaration *decl);

/// Create a null DValue.
DValue *DtoNullValue(Type *t, Loc loc = Loc());

// casts
DValue *DtoCastInt(Loc &loc, DValue *val, Type *to);
DValue *DtoCastPtr(Loc &loc, DValue *val, Type *to);
DValue *DtoCastFloat(Loc &loc, DValue *val, Type *to);
DValue *DtoCastDelegate(Loc &loc, DValue *val, Type *to);
DValue *DtoCast(Loc &loc, DValue *val, Type *to);

// return the same val as passed in, modified to the target type, if possible,
// otherwise returns a new DValue
DValue *DtoPaintType(Loc &loc, DValue *val, Type *to);

// is template instance check, returns module where instantiated
TemplateInstance *DtoIsTemplateInstance(Dsymbol *s);

/// Makes sure the declarations corresponding to the given D symbol have been
/// emitted to the currently processed LLVM module.
///
/// This means that dsym->ir can be expected to set to reasonable values.
///
/// This function does *not* emit any (function, variable) *definitions*; this
/// is done by Dsymbol::codegen.
void DtoResolveDsymbol(Dsymbol *dsym);
void DtoResolveVariable(VarDeclaration *var);

// declaration inside a declarationexp
void DtoVarDeclaration(VarDeclaration *var);
DValue *DtoDeclarationExp(Dsymbol *declaration);
LLValue *DtoRawVarDeclaration(VarDeclaration *var, LLValue *addr = nullptr);

// initializer helpers
LLConstant *DtoConstInitializer(Loc &loc, Type *type,
                                Initializer *init = nullptr);
LLConstant *DtoConstExpInit(Loc &loc, Type *targetType, Expression *exp);

// getting typeinfo of type, base=true casts to object.TypeInfo
LLConstant *DtoTypeInfoOf(Type *ty, bool base = true);

// target stuff
void findDefaultTarget();

/// Returns a pointer to the given member field of an aggregate.
///
/// 'src' is a pointer to the start of the memory of an 'ad' instance.
LLValue *DtoIndexAggregate(LLValue *src, AggregateDeclaration *ad,
                           VarDeclaration *vd);

/// Returns the index of a given member variable in the resulting LLVM type of
/// an aggregate.
///
/// This is only a valid operation if the field is known to be non-overlapping,
/// so that no byte-wise offset is needed.
unsigned getFieldGEPIndex(AggregateDeclaration *ad, VarDeclaration *vd);

///
DValue *DtoInlineAsmExpr(Loc &loc, FuncDeclaration *fd, Expressions *arguments,
                         LLValue *sretPointer = nullptr);
///
llvm::CallInst *DtoInlineAsmExpr(const Loc &loc, llvm::StringRef code,
                                 llvm::StringRef constraints,
                                 llvm::ArrayRef<llvm::Value *> operands,
                                 llvm::Type *returnType);

/// Returns the size the LLVM type for a member variable of the given type will
/// take up in a struct (in bytes). This does not include padding in any way.
size_t getMemberSize(Type *type);

/// Returns the llvm::Value of the passed DValue, making sure that it is an
/// lvalue (has a memory address), so it can be passed to the D runtime
/// functions without problems.
LLValue *makeLValue(Loc &loc, DValue *value);

void callPostblit(Loc &loc, Expression *exp, LLValue *val);

/// Returns whether the given variable is a DMD-internal "ref variable".
///
/// D doesn't have reference variables (the ref keyword is only usable in
/// function signatures and foreach headers), but the DMD frontend internally
/// creates them in cases like lowering a ref foreach to a for loop or the
/// implicit __result variable for ref-return functions with out contracts.
bool isSpecialRefVar(VarDeclaration *vd);

/// Returns whether the type is unsigned in LLVM terms, which also includes
/// pointers.
bool isLLVMUnsigned(Type *t);

/// Converts a DMD comparison operation token into the corresponding LLVM icmp
/// predicate for the given operand signedness.
///
/// For some operations, the result can be a constant. In this case outConst is
/// set to it, otherwise outPred is set to the predicate to use.
void tokToICmpPred(TOK op, bool isUnsigned, llvm::ICmpInst::Predicate *outPred,
                   llvm::Value **outConst);

/// Converts a DMD equality/identity operation token into the corresponding LLVM
/// icmp predicate.
llvm::ICmpInst::Predicate eqTokToICmpPred(TOK op, bool invert = false);

/// For equality/identity operations, returns `(lhs1 == rhs1) & (lhs2 == rhs2)`.
/// `(lhs1 != rhs1) | (lhs2 != rhs2)` for inequality/not-identity.
LLValue *createIPairCmp(TOK op, LLValue *lhs1, LLValue *lhs2, LLValue *rhs1,
                        LLValue *rhs2);

////////////////////////////////////////////
// gen/tocall.cpp stuff below
////////////////////////////////////////////

///
IrFuncTy &DtoIrTypeFunction(DValue *fnval);
///
TypeFunction *DtoTypeFunction(DValue *fnval);

///
LLValue *DtoCallableValue(DValue *fn);

///
LLFunctionType *DtoExtractFunctionType(LLType *type);

/// Checks whether fndecl is an intrinsic that requires special lowering. If so,
/// emits the code for it and returns true, settings result to the resulting
/// DValue (if any). If the call does not correspond to a "magic" intrinsic,
/// i.e. should be turned into a normal function call, returns false.
bool DtoLowerMagicIntrinsic(IRState *p, FuncDeclaration *fndecl, CallExp *e,
                            DValue *&result);

///
DValue *DtoCallFunction(Loc &loc, Type *resulttype, DValue *fnval,
                        Expressions *arguments, LLValue *sretPointer = nullptr);

Type *stripModifiers(Type *type, bool transitive = false);

void printLabelName(std::ostream &target, const char *func_mangle,
                    const char *label_name);

void AppendFunctionToLLVMGlobalCtorsDtors(llvm::Function *func,
                                          const uint32_t priority,
                                          const bool isCtor);

template <typename T>
LLConstant *toConstantArray(LLType *ct, LLArrayType *at, T *str, size_t len,
                            bool nullterm = true) {
  std::vector<LLConstant *> vals;
  vals.reserve(len + 1);
  for (size_t i = 0; i < len; ++i) {
    vals.push_back(LLConstantInt::get(ct, str[i], false));
  }
  if (nullterm) {
    vals.push_back(LLConstantInt::get(ct, 0, false));
  }
  return LLConstantArray::get(at, vals);
}

llvm::Constant *buildStringLiteralConstant(StringExp *se, bool zeroTerm);

/// Tries to declare an LLVM global. If a variable with the same mangled name
/// already exists, checks if the types match and returns it instead.
///
/// Necessary to support multiple declarations with the same mangled name, as
/// can be the case due to pragma(mangle).
llvm::GlobalVariable *declareGlobal(const Loc &loc, llvm::Module &module,
                                    llvm::Type *type,
                                    llvm::StringRef mangledName,
                                    bool isConstant,
                                    bool isThreadLocal = false);

/// Defines an existing LLVM global, i.e., sets the initial value and finalizes
/// its linkage and visibility.
/// Asserts that a global isn't defined multiple times this way.
void defineGlobal(llvm::GlobalVariable *global, llvm::Constant *init,
                  Dsymbol *symbolForLinkageAndVisibility);

/// Declares (if not already declared) & defines an LLVM global.
llvm::GlobalVariable *defineGlobal(const Loc &loc, llvm::Module &module,
                                   llvm::StringRef mangledName,
                                   llvm::Constant *init,
                                   llvm::GlobalValue::LinkageTypes linkage,
                                   bool isConstant, bool isThreadLocal = false);

FuncDeclaration *getParentFunc(Dsymbol *sym);

void Declaration_codegen(Dsymbol *decl);
void Declaration_codegen(Dsymbol *decl, IRState *irs);

DValue *toElem(Expression *e);
/// If `skipOverCasts` is true, skips over casts (no codegen) and returns the
/// (casted) result of the first inner non-cast expression.
DValue *toElem(Expression *e, bool skipOverCasts);
DValue *toElemDtor(Expression *e);
LLConstant *toConstElem(Expression *e, IRState *p);

inline llvm::Value *DtoRVal(Expression *e) { return DtoRVal(toElem(e)); }
inline llvm::Value *DtoLVal(Expression *e) { return DtoLVal(toElem(e)); }

/// Creates a DLValue for the given VarDeclaration.
///
/// If the storage is not given explicitly, the declaration is expected to be
/// already resolved, and the value from the associated IrVar will be used.
DValue *makeVarDValue(Type *type, VarDeclaration *vd,
                      llvm::Value *storage = nullptr);

/// Checks whether the rhs expression is able to construct the lhs lvalue
/// directly in-place. If so, it performs the according codegen and returns
/// true; otherwise it just returns false.
bool toInPlaceConstruction(DLValue *lhs, Expression *rhs);

std::string llvmTypeToString(LLType *type);
