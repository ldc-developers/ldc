//===-- cl_options-llvm.cpp -----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options-llvm.h"

#if LDC_WITH_LLD
#include "llvm/ADT/Triple.h"
#endif

// Pull in command-line options and helper functions from special LLVM header
// shared by multiple LLVM tools.
#include "llvm/CodeGen/CommandFlags.h"
static llvm::codegen::RegisterCodeGenFlags CGF;
using namespace llvm;

static cl::opt<bool>
    DisableRedZone("disable-red-zone", cl::ZeroOrMore,
                   cl::desc("Do not emit code that uses the red zone."));

// Now expose the helper functions (with static linkage) via external wrappers
// in the opts namespace, including some additional helper functions.
namespace opts {

std::string getArchStr() {
  return codegen::getMArch();
}

Optional<Reloc::Model> getRelocModel() {
  return codegen::getExplicitRelocModel();
}

Optional<CodeModel::Model> getCodeModel() {
  return codegen::getExplicitCodeModel();
}

#if LDC_LLVM_VER >= 1300
using FPK = llvm::FramePointerKind;
#else
using FPK = llvm::FramePointer::FP;
#endif

llvm::Optional<FPK> framePointerUsage() {
  // Defaults to `FP::None`; no way to check if set explicitly by user except
  // indirectly via setFunctionAttributes()...
  return codegen::getFramePointerUsage();
}

bool disableRedZone() { return ::DisableRedZone; }

bool printTargetFeaturesHelp() {
  const auto MCPU = codegen::getMCPU();
  const auto MAttrs = codegen::getMAttrs();
  if (MCPU == "help")
    return true;
  return std::any_of(MAttrs.begin(), MAttrs.end(),
                     [](const std::string &a) { return a == "help"; });
}

TargetOptions InitTargetOptionsFromCodeGenFlags(const llvm::Triple &triple) {
#if LDC_LLVM_VER >= 1200
  return codegen::InitTargetOptionsFromCodeGenFlags(triple);
#else
  return codegen::InitTargetOptionsFromCodeGenFlags();
#endif
}

std::string getCPUStr() {
  return codegen::getCPUStr();
}

std::string getFeaturesStr() {
  return codegen::getFeaturesStr();
}

void setFunctionAttributes(StringRef cpu, StringRef features,
                           Function &function) {
  return codegen::setFunctionAttributes(cpu, features, function);
}
} // namespace opts

#if LDC_WITH_LLD
// LLD uses the shared header too (for LTO) and exposes some wrappers in
// the lld namespace. Define them here to prevent the LLD object from being
// linked in with its conflicting command-line options.
namespace lld {
TargetOptions initTargetOptionsFromCodeGenFlags() {
  return ::opts::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
}

Optional<Reloc::Model> getRelocModelFromCMModel() {
  return ::opts::getRelocModel();
}

Optional<CodeModel::Model> getCodeModelFromCMModel() {
  return ::opts::getCodeModel();
}

std::string getCPUStr() { return ::opts::getCPUStr(); }

std::vector<std::string> getMAttrs() { return codegen::getMAttrs(); }
} // namespace lld
#endif // LDC_WITH_LLD
