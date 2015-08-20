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

#ifndef LDC_GEN_IRSTATE_H
#define LDC_GEN_IRSTATE_H

#include "aggregate.h"
#include "root.h"
#include "ir/iraggr.h"
#include "ir/irvar.h"
#include "gen/dibuilder.h"
#include <deque>
#include <list>
#include <set>
#include <sstream>
#include <vector>
#include "llvm/ADT/StringMap.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/CallSite.h"
#else
#include "llvm/Support/CallSite.h"
#endif

namespace llvm {
    class LLVMContext;
    class TargetMachine;
}

// global ir state for current module
struct IRState;
struct TargetABI;

extern IRState* gIR;
extern llvm::TargetMachine* gTargetMachine;
#if LDC_LLVM_VER >= 302
extern const llvm::DataLayout* gDataLayout;
#else
extern const llvm::TargetData* gDataLayout;
#endif
extern TargetABI* gABI;

class TypeFunction;
class TypeStruct;
class ClassDeclaration;
class FuncDeclaration;
class Module;
class TypeStruct;
struct BaseClass;
class AnonDeclaration;

struct IrFunction;
struct IrModule;

// represents a scope
struct IRScope
{
    llvm::BasicBlock* begin;
    IRBuilder<> builder;

    IRScope();
    explicit IRScope(llvm::BasicBlock* b);

    const IRScope& operator=(const IRScope& rhs);
};

struct IRBuilderHelper
{
    IRState* state;
    IRBuilder<>* operator->();
};

struct IRAsmStmt
{
    IRAsmStmt()
    : isBranchToLabel(NULL) {}

    std::string code;
    std::string out_c;
    std::string in_c;
    std::vector<LLValue*> out;
    std::vector<LLValue*> in;

    // if this is nonzero, it contains the target label
    LabelDsymbol* isBranchToLabel;
};

struct IRAsmBlock
{
    std::deque<IRAsmStmt*> s;
    std::set<std::string> clobs;
    size_t outputcount;

    // stores the labels within the asm block
    std::vector<Identifier*> internalLabels;

    CompoundAsmStatement* asmBlock;
    LLType* retty;
    unsigned retn;
    bool retemu; // emulate abi ret with a temporary
    LLValue* (*retfixup)(IRBuilderHelper b, LLValue* orig); // Modifies retval

    IRAsmBlock(CompoundAsmStatement* b)
        : outputcount(0), asmBlock(b), retty(NULL), retn(0), retemu(false),
        retfixup(NULL)
    {}
};

// represents the module
struct IRState
{
    IRState(const char *name, llvm::LLVMContext &context);

    llvm::Module module;
    llvm::LLVMContext& context() const { return module.getContext(); }

    Module *dmodule;

    LLStructType* mutexType;
    LLStructType* moduleRefType;

    // functions
    typedef std::vector<IrFunction*> FunctionVector;
    FunctionVector functions;
    IrFunction* func();

    llvm::Function* topfunc();
    llvm::Instruction* topallocapoint();

    // The function containing the D main() body, if any (not the actual main()
    // implicitly emitted).
    llvm::Function* mainFunc;

    // basic block scopes
    std::vector<IRScope> scopes;
    IRScope& scope();
    llvm::BasicBlock* scopebb();
    bool scopereturned();

    // create a call or invoke, depending on the landing pad info
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, LLValue* Arg3, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, LLValue* Arg2,  LLValue* Arg3, LLValue* Arg4, const char* Name="");

    // this holds the array being indexed or sliced so $ will work
    // might be a better way but it works. problem is I only get a
    // VarDeclaration for __dollar, but I can't see how to get the
    // array pointer from this :(
    std::vector<DValue*> arrays;

    // builder helper
    IRBuilderHelper ir;

    // debug info helper
    ldc::DIBuilder DBuilder;

    // for inline asm
    IRAsmBlock* asmBlock;
    std::ostringstream nakedAsm;

    // Globals to pin in the llvm.used array to make sure they are not
    // eliminated.
    std::vector<LLConstant*> usedArray;

    /// Whether to emit array bounds checking in the current function.
    bool emitArrayBoundsChecks();

    // Global variables bound to string literals.  Once created such a
    // variable is reused whenever the same string literal is
    // referenced in the module.  Caching them per module prevents the
    // duplication of identical literals.
    llvm::StringMap<llvm::GlobalVariable*> stringLiteral1ByteCache;
    llvm::StringMap<llvm::GlobalVariable*> stringLiteral2ByteCache;
    llvm::StringMap<llvm::GlobalVariable*> stringLiteral4ByteCache;

#if LDC_LLVM_VER >= 303
    /// Vector of options passed to the linker as metadata in object file.
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata *, 5> LinkerMetadataArgs;
#else
    llvm::SmallVector<llvm::Value *, 5> LinkerMetadataArgs;
#endif
#endif
};

void Statement_toIR(Statement *s, IRState *irs);

#endif // LDC_GEN_IRSTATE_H
