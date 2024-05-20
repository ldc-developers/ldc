//===-- gen/irstate.h - Global codegen state --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the global state used and modified when generating the
// code (i.e. LLVM IR) for a given D module.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/aggregate.h"
#include "gen/dibuilder.h"
#include "gen/objcgen.h"
#include "ir/iraggr.h"
#include "ir/irvar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include <deque>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

namespace llvm {
class LLVMContext;
class TargetMachine;
class IndexedInstrProfReader;
}

class FuncGenState;
struct IRState;
struct TargetABI;
class DComputeTarget;

extern IRState *gIR;
extern llvm::TargetMachine *gTargetMachine;
extern const llvm::DataLayout *gDataLayout;
extern TargetABI *gABI;

class TypeFunction;
class TypeStruct;
class ClassDeclaration;
class FuncDeclaration;
class Module;
class TypeStruct;
struct BaseClass;
class AnonDeclaration;
class StructLiteralExp;

struct IrFunction;
struct IrModule;

// Saves the IRBuilder state and restores it on destruction.
struct IRBuilderScope {
private:
  llvm::IRBuilderBase::InsertPointGuard ipGuard;
  llvm::IRBuilderBase::FastMathFlagGuard fmfGuard;

public:
  explicit IRBuilderScope(llvm::IRBuilderBase &builder)
      : ipGuard(builder), fmfGuard(builder) {}
};

struct IRBuilderHelper {
  IRState *state;
  IRBuilder<> *operator->();
};

struct IRAsmStmt {
  IRAsmStmt() : isBranchToLabel(nullptr) {}

  std::string code;
  struct Operands {
    std::string c; // contraint
    std::vector<LLValue *> ops;
    std::vector<Type *> dTypes;
  };
  Operands out, in;

  // if this is nonzero, it contains the target label
  LabelDsymbol *isBranchToLabel;
};

struct IRAsmBlock {
  std::deque<IRAsmStmt *> s;
  std::set<std::string> clobs;
  size_t outputcount;

  // stores the labels within the asm block
  std::vector<Identifier *> internalLabels;

  CompoundAsmStatement *asmBlock;
  LLType *retty;
  unsigned retn;
  bool retemu; // emulate abi ret with a temporary
  LLValue *(*retfixup)(IRBuilderHelper b, LLValue *orig); // Modifies retval

  explicit IRAsmBlock(CompoundAsmStatement *b)
      : outputcount(0), asmBlock(b), retty(nullptr), retn(0), retemu(false),
        retfixup(nullptr) {}
};

// represents the LLVM module (object file)
struct IRState {
private:
  IRBuilder<> builder;
  friend struct IRBuilderHelper;

  std::vector<std::pair<llvm::GlobalVariable *, llvm::Constant *>>
      globalsToReplace;
  Array<Loc> inlineAsmLocs; // tracked by GC

  // Cache of global variables for taking the address of struct literal
  // constants. (Also) used to resolve self-references. Must be cached per IR
  // module: https://github.com/ldc-developers/ldc/issues/2990
  // [The real key type is `StructLiteralExp *`; a fwd class declaration isn't
  // enough to use it directly.]
  llvm::DenseMap<void *, llvm::GlobalVariable *> structLiteralGlobals;

  // Global variables bound to string literals. Once created such a variable
  // is reused whenever an equivalent string literal is referenced in the
  // module, to prevent duplicates.
  llvm::StringMap<llvm::GlobalVariable *> cachedStringLiterals;
  llvm::StringMap<llvm::GlobalVariable *> cachedWstringLiterals;
  llvm::StringMap<llvm::GlobalVariable *> cachedDstringLiterals;

public:
  IRState(const char *name, llvm::LLVMContext &context);
  ~IRState();

  IRState(IRState const &) = delete;
  IRState &operator=(IRState const &) = delete;

  llvm::Module module;
  llvm::LLVMContext &context() const { return module.getContext(); }

  Module *dmodule = nullptr;

  LLStructType *moduleRefType = nullptr;

  ObjCState objc;

  // Stack of currently codegen'd functions (more than one for lambdas or other
  // nested functions, inlining-only codegen'ing, -linkonce-templates etc.),
  // and some convenience accessors for the top-most one.
  std::vector<std::unique_ptr<FuncGenState>> funcGenStates;
  FuncGenState &funcGen();
  IrFunction *func();
  llvm::Function *topfunc();
  llvm::Instruction *topallocapoint();

  // Use this to set the IRBuilder's insertion point for a new function.
  // The previous IRBuilder state is restored when the returned value is
  // destructed. Use `ir->SetInsertPoint()` instead to change the insertion
  // point inside the same function.
  std::unique_ptr<IRBuilderScope> setInsertPoint(llvm::BasicBlock *bb);
  // Use this to have the IRBuilder's current insertion point (incl. debug
  // location) restored when the returned value is destructed.
  std::unique_ptr<llvm::IRBuilderBase::InsertPointGuard> saveInsertPoint();
  // Returns the basic block the IRBuilder currently inserts into.
  llvm::BasicBlock *scopebb() { return ir->GetInsertBlock(); }
  bool scopereturned();

  // Creates a new basic block and inserts it before the specified one.
  llvm::BasicBlock *insertBBBefore(llvm::BasicBlock *successor,
                                   const llvm::Twine &name);
  // Creates a new basic block and inserts it after the specified one.
  llvm::BasicBlock *insertBBAfter(llvm::BasicBlock *predecessor,
                                  const llvm::Twine &name);
  // Creates a new basic block and inserts it after the current scope basic
  // block (`scopebb()`).
  llvm::BasicBlock *insertBB(const llvm::Twine &name);

  // create a call or invoke, depending on the landing pad info
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee,
                                        const char *Name = "");
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee,
                                        llvm::ArrayRef<LLValue *> Args,
                                        const char *Name = "",
                                        bool isNothrow = false);
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee, LLValue *Arg1,
                                        const char *Name = "");
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee, LLValue *Arg1,
                                        LLValue *Arg2, const char *Name = "");
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee, LLValue *Arg1,
                                        LLValue *Arg2, LLValue *Arg3,
                                        const char *Name = "");
  llvm::Instruction *CreateCallOrInvoke(LLFunction *Callee, LLValue *Arg1,
                                        LLValue *Arg2, LLValue *Arg3,
                                        LLValue *Arg4, const char *Name = "");

  // this holds the array being indexed or sliced so $ will work
  // might be a better way but it works. problem is I only get a
  // VarDeclaration for __dollar, but I can't see how to get the
  // array pointer from this :(
  std::vector<DValue *> arrays;

  // builder helper
  IRBuilderHelper ir;

  // debug info helper
  ldc::DIBuilder DBuilder;

  // PGO data file reader
  std::unique_ptr<llvm::IndexedInstrProfReader> PGOReader;
  llvm::IndexedInstrProfReader *getPGOReader() const { return PGOReader.get(); }

  // for inline asm
  IRAsmBlock *asmBlock = nullptr;
  std::ostringstream nakedAsm;

  // Globals to pin in the llvm.used array to make sure they are not
  // eliminated.
  std::vector<LLConstant *> usedArray;

  /// Whether to emit array bounds checking in the current function.
  bool emitArrayBoundsChecks();

  // Sets the initializer for a global LL variable.
  // If the types don't match, this entails creating a new helper global
  // matching the initializer type and replacing all existing uses of globalVar
  // by the new helper global.
  // Returns either the specified globalVar if the types match, or the new
  // helper global replacing globalVar.
  llvm::GlobalVariable *
  setGlobalVarInitializer(llvm::GlobalVariable *globalVar,
                          llvm::Constant *initializer,
                          Dsymbol *symbolForLinkageAndVisibility);

  // To be called when finalizing the IR module in order to perform a second
  // replacement pass for global variables replaced (and registered) by
  // setGlobalVarInitializer().
  void replaceGlobals();

  llvm::GlobalVariable *getStructLiteralGlobal(StructLiteralExp *sle) const;
  void setStructLiteralGlobal(StructLiteralExp *sle,
                              llvm::GlobalVariable *global);

  // Constructs a global variable for a StringExp.
  // Caches the result based on StringExp::peekData() such that any subsequent
  // calls with a StringExp with matching data will return the same variable.
  // Exception: ulong[]-typed hex strings (not null-terminated either).
  llvm::GlobalVariable *getCachedStringLiteral(StringExp *se);
  llvm::GlobalVariable *getCachedStringLiteral(llvm::StringRef s);

  // List of functions with cpu or features attributes overriden by user
  std::vector<IrFunction *> targetCpuOrFeaturesOverridden;

  struct RtCompiledFuncDesc {
    llvm::GlobalVariable *thunkVar;
    llvm::Function *thunkFunc;
  };

  std::map<llvm::Function *, RtCompiledFuncDesc> dynamicCompiledFunctions;
  std::set<IrGlobal *> dynamicCompiledVars;

/// Vector of options passed to the linker as metadata in object file.
  llvm::SmallVector<llvm::MDNode *, 5> linkerOptions;
  llvm::SmallVector<llvm::MDNode *, 5> linkerDependentLibs;

  void addLinkerOption(llvm::ArrayRef<llvm::StringRef> options);
  void addLinkerDependentLib(llvm::StringRef libraryName);

  llvm::CallInst *createInlineAsmCall(const Loc &loc, llvm::InlineAsm *ia,
                                      llvm::ArrayRef<llvm::Value *> args,
                                      llvm::ArrayRef<llvm::Type *> indirectTypes);
  void addInlineAsmSrcLoc(const Loc &loc, llvm::CallInst *inlineAsmCall);
  const Loc &getInlineAsmSrcLoc(unsigned srcLocCookie) const;

  // MS C++ compatible type descriptors
  llvm::DenseMap<size_t, llvm::StructType *> TypeDescriptorTypeMap;
  llvm::DenseMap<ClassDeclaration *, llvm::GlobalVariable *> TypeDescriptorMap;

  // Target for dcompute. If not nullptr, it owns this.
  DComputeTarget *dcomputetarget = nullptr;
};

void Statement_toIR(Statement *s, IRState *irs);

bool useMSVCEH();
