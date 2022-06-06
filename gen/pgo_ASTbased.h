//===-- gen/pgo_ASTbased.h - Code Coverage Analysis -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is adapted from CodeGenPGO.h (Clang, LLVM). Therefore,
// this file is distributed under the LLVM license.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to generate instrumentation code for
// AST-based profile-guided optimization.
// PGO is enabled by compiling first with "-fprofile-instr-generate",
// and then with "-fprofile-instr-use=filename.profdata".
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/llvm.h"
#include "llvm/ProfileData/InstrProf.h"
#include <string>
#include <vector>
#include <array>

namespace llvm {
class GlobalVariable;
class Function;
class IndexedInstrProfReader;
}
class FuncDeclaration;
struct IRState;
class RootObject;
class ForStatement;
class ForeachStatement;
class ForeachRangeStatement;


/// Keeps per-function PGO state.
class CodeGenPGO {
public:
  CodeGenPGO()
      : NumRegionCounters(0), FunctionHash(0), CurrentRegionCount(0),
        NumValueSites({{0}}) {}

  /// Whether or not we emit PGO instrumentation for the current function.
  bool emitsInstrumentation() const { return emitInstrumentation; }

  /// Whether or not we have PGO region data for the current function. This is
  /// false both when we have no data at all and when our data has been
  /// discarded.
  bool haveRegionCounts() const { return !RegionCounts.empty(); }

  /// Return the counter value of the current region.
  uint64_t getCurrentRegionCount() const { return CurrentRegionCount; }

  /// Set the counter value for the current region. This is used to keep track
  /// of changes to the most recent counter from control flow and non-local
  /// exits.
  void setCurrentRegionCount(uint64_t Count) { CurrentRegionCount = Count; }

  /// If the execution count for the current statement is known, record that
  /// as the current count.
  /// Return the current count.
  uint64_t setCurrentStmt(const RootObject *S) {
    auto Count = getStmtCount(S);
    if (Count.first)
      setCurrentRegionCount(Count.second);

    return CurrentRegionCount;
  }

  /// Check if we need to emit coverage mapping for a given declaration
  //  void checkGlobalDecl(GlobalDecl GD);

  /// Assign counters to regions and configure them for PGO of a given
  /// function. Does nothing if instrumentation is not enabled and either
  /// generates global variables or associates PGO data with each of the
  /// counters depending on whether we are generating or using instrumentation.
  void assignRegionCounters(const FuncDeclaration *D, llvm::Function *Fn);

  /// Emit a coverage mapping range with a counter zero
  /// for an unused declaration.
  //  void emitEmptyCounterMapping(const Decl *D, StringRef FuncName,
  //                               llvm::GlobalValue::LinkageTypes Linkage);

  void emitCounterIncrement(const RootObject *S) const;

  /// Return the region count for the counter at the given index.
  uint64_t getRegionCount(const RootObject *S) const {
    if (!RegionCounterMap)
      return 0;
    if (!haveRegionCounts())
      return 0;
    return RegionCounts[(*RegionCounterMap)[S]];
  }

  llvm::MDNode *createProfileWeights(uint64_t TrueCount,
                                     uint64_t FalseCount) const;
  llvm::MDNode *createProfileWeights(llvm::ArrayRef<uint64_t> Weights) const;
  llvm::MDNode *createProfileWeightsWhileLoop(const RootObject *Cond,
                                              uint64_t LoopCount) const;
  llvm::MDNode *createProfileWeightsForLoop(const ForStatement *stmt) const;
  llvm::MDNode *createProfileWeightsForeach(const ForeachStatement *stmt) const;
  llvm::MDNode *
  createProfileWeightsForeachRange(const ForeachRangeStatement *stmt) const;

  /// Get counter associated with RootObject pointer.
  static RootObject *getCounterPtr(const RootObject *ptr, unsigned counter_idx);

  /// Apply branch weights to instruction (br or switch)
  template <typename InstTy>
  static InstTy *addBranchWeights(InstTy *I, llvm::MDNode *weights) {
    if (weights)
      I->setMetadata(llvm::LLVMContext::MD_prof, weights);
    return I;
  }

  /// Adds profiling instrumentation/annotation of indirect calls to `funcPtr`
  /// for callsite `callSite`.
  void emitIndirectCallPGO(llvm::Instruction *callSite, llvm::Value *funcPtr);

  /// Adds profiling instrumentation/annotation of a certain value.
  /// This method either inserts a call to the profile run-time during
  /// instrumentation or puts profile data into metadata for PGO use.
  /// The profiled value is of kind `valueKind`, will be added right before IR
  /// code site `valueSite`, and the to be profiled value is given by
  /// `value`. `value` should be of LLVM i64 type, unless `ptrCastNeeded` is
  /// true, in which case a ptrtoint cast to i64 is added.
  void valueProfile(uint32_t valueKind, llvm::Instruction *valueSite,
                    llvm::Value *value, bool ptrCastNeeded);

private:
  std::string FuncName;
  llvm::GlobalVariable *FuncNameVar;

  unsigned NumRegionCounters;
  uint64_t FunctionHash;
  std::unique_ptr<llvm::DenseMap<const RootObject *, unsigned>>
      RegionCounterMap;
  std::unique_ptr<llvm::DenseMap<const RootObject *, uint64_t>> StmtCountMap;
  std::vector<uint64_t> RegionCounts;
  uint64_t CurrentRegionCount;

  std::array<unsigned, llvm::IPVK_Last + 1> NumValueSites;
  std::unique_ptr<llvm::InstrProfRecord> ProfRecord;

  /// \brief A flag that is set to false when instrumentation code should not be
  /// emitted for this function.
  bool emitInstrumentation = true;

  /// Check if an execution count is known for a given statement. If so, return
  /// true and put the value in pair::second; else return false.
  std::pair<bool, uint64_t> getStmtCount(const RootObject *S) const {
    if (!StmtCountMap)
      return std::make_pair(false, 0);
    auto I = StmtCountMap->find(S);
    if (I == StmtCountMap->end())
      return std::make_pair(false, 0);
    return std::make_pair(true, I->second);
  }

  void setFuncName(llvm::Function *Fn);
  void setFuncName(llvm::StringRef Name,
                   llvm::GlobalValue::LinkageTypes Linkage);
  void mapRegionCounters(const FuncDeclaration *D);
  void computeRegionCounts(const FuncDeclaration *D);
  void applyFunctionAttributes(llvm::Function *Fn);
  void loadRegionCounts(llvm::IndexedInstrProfReader *PGOReader,
                        const FuncDeclaration *D);
};
