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
#include "dmd/root/root.h"
#include "gen/dibuilder.h"
#include "gen/objcgen.h"
#include "ir/iraggr.h"
#include "ir/irvar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/CallSite.h"
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

// represents a scope
struct IRScope {
  llvm::BasicBlock *begin;
  IRBuilder<> builder;

  IRScope();
  IRScope(const IRScope &) = default;
  explicit IRScope(llvm::BasicBlock *b);

  IRScope &operator=(const IRScope &rhs);
};

struct IRBuilderHelper {
  IRState *state;
  IRBuilder<> *operator->();
};

struct IRAsmStmt {
  IRAsmStmt() : isBranchToLabel(nullptr) {}

  std::string code;
  std::string out_c;
  std::string in_c;
  std::vector<LLValue *> out;
  std::vector<LLValue *> in;

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
  std::vector<std::pair<llvm::GlobalVariable *, llvm::Constant *>>
      globalsToReplace;
  Array<Loc> inlineAsmLocs; // tracked by GC

  // Cache of (possibly bitcast) global variables for taking the address of
  // struct literal constants. (Also) used to resolve self-references. Must be
  // cached per IR module: https://github.com/ldc-developers/ldc/issues/2990
  // [The real key type is `StructLiteralExp *`; a fwd class declaration isn't
  // enough to use it directly.]
  llvm::DenseMap<void *, llvm::Constant *> structLiteralConstants;

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
  // nested functions, inlining-only codegen'ing, etc.), and some convenience
  // accessors for the top-most one.
  std::vector<std::unique_ptr<FuncGenState>> funcGenStates;
  FuncGenState &funcGen();
  IrFunction *func();
  llvm::Function *topfunc();
  llvm::Instruction *topallocapoint();

  // The function containing the D main() body, if any (not the actual main()
  // implicitly emitted).
  llvm::Function *mainFunc = nullptr;

  // basic block scopes
  std::vector<IRScope> scopes;
  IRScope &scope();
  llvm::BasicBlock *scopebb();
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
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee, const char *Name = "");
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee,
                                    llvm::ArrayRef<LLValue *> Args,
                                    const char *Name = "",
                                    bool isNothrow = false);
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                    const char *Name = "");
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                    LLValue *Arg2, const char *Name = "");
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                    LLValue *Arg2, LLValue *Arg3,
                                    const char *Name = "");
  llvm::CallSite CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                    LLValue *Arg2, LLValue *Arg3, LLValue *Arg4,
                                    const char *Name = "");

  bool isMainFunc(const IrFunction *func) const;

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

  // Global variables bound to string literals.  Once created such a
  // variable is reused whenever the same string literal is
  // referenced in the module.  Caching them per module prevents the
  // duplication of identical literals.
  llvm::StringMap<llvm::GlobalVariable *> stringLiteral1ByteCache;
  llvm::StringMap<llvm::GlobalVariable *> stringLiteral2ByteCache;
  llvm::StringMap<llvm::GlobalVariable *> stringLiteral4ByteCache;

  // Sets the initializer for a global LL variable.
  // If the types don't match, this entails creating a new helper global
  // matching the initializer type and replacing all existing uses of globalVar
  // by a bitcast pointer to the helper global's payload.
  // Returns either the specified globalVar if the types match, or the bitcast
  // pointer replacing globalVar (and resets globalVar to the new helper
  // global).
  llvm::Constant *setGlobalVarInitializer(llvm::GlobalVariable *&globalVar,
                                          llvm::Constant *initializer);

  // To be called when finalizing the IR module in order to perform a second
  // replacement pass for global variables replaced (and registered) by
  // setGlobalVarInitializer().
  void replaceGlobals();

  llvm::Constant *getStructLiteralConstant(StructLiteralExp *sle) const;
  void setStructLiteralConstant(StructLiteralExp *sle,
                                llvm::Constant *constant);

  // List of functions with cpu or features attributes overriden by user
  std::vector<IrFunction *> targetCpuOrFeaturesOverridden;

  struct RtCompiledFuncDesc {
    llvm::GlobalVariable *thunkVar;
    llvm::Function *thunkFunc;
  };

  std::map<llvm::Function *, RtCompiledFuncDesc> dynamicCompiledFunctions;
  std::set<IrGlobal *> dynamicCompiledVars;

/// Vector of options passed to the linker as metadata in object file.
#if LDC_LLVM_VER >= 500
  llvm::SmallVector<llvm::MDNode *, 5> linkerOptions;
  llvm::SmallVector<llvm::MDNode *, 5> linkerDependentLibs;
#else
  llvm::SmallVector<llvm::Metadata *, 5> linkerOptions;
  llvm::SmallVector<llvm::Metadata *, 5> linkerDependentLibs;
#endif

  void addLinkerOption(llvm::ArrayRef<llvm::StringRef> options);
  void addLinkerDependentLib(llvm::StringRef libraryName);

  void addInlineAsmSrcLoc(const Loc &loc, llvm::CallInst *inlineAsmCall);
  const Loc &getInlineAsmSrcLoc(unsigned srcLocCookie) const;

  // MS C++ compatible type descriptors
  llvm::DenseMap<size_t, llvm::StructType *> TypeDescriptorTypeMap;
  llvm::DenseMap<llvm::Constant *, llvm::GlobalVariable *> TypeDescriptorMap;

  // Target for dcompute. If not nullptr, it owns this.
  DComputeTarget *dcomputetarget = nullptr;
};

void Statement_toIR(Statement *s, IRState *irs);

bool useMSVCEH();
