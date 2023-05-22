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
#if LDC_LLVM_VER >= 1100
#include "llvm/CodeGen/CommandFlags.h"
static llvm::codegen::RegisterCodeGenFlags CGF;
using namespace llvm;
#else
#include "llvm/CodeGen/CommandFlags.inc"
#endif

static cl::opt<bool>
    DisableRedZone("disable-red-zone", cl::ZeroOrMore,
                   cl::desc("Do not emit code that uses the red zone."));

#if LDC_LLVM_VER < 1100
// legacy option
static cl::opt<bool>
    disableFPElim("disable-fp-elim", cl::ZeroOrMore, cl::ReallyHidden,
                  cl::desc("Disable frame pointer elimination optimization"));
#endif

// Now expose the helper functions (with static linkage) via external wrappers
// in the opts namespace, including some additional helper functions.
namespace opts {

std::string getArchStr() {
#if LDC_LLVM_VER >= 1100
  return codegen::getMArch();
#else
  return ::MArch;
#endif
}

Optional<Reloc::Model> getRelocModel() {
#if LDC_LLVM_VER >= 1100
  return codegen::getExplicitRelocModel();
#else
  return ::getRelocModel();
#endif
}

Optional<CodeModel::Model> getCodeModel() {
#if LDC_LLVM_VER >= 1100
  return codegen::getExplicitCodeModel();
#else
  return ::getCodeModel();
#endif
}

#if LDC_LLVM_VER >= 1300
using FPK = llvm::FramePointerKind;
#else
using FPK = llvm::FramePointer::FP;
#endif

std::optional<FPK> framePointerUsage() {
#if LDC_LLVM_VER >= 1100
  // Defaults to `FP::None`; no way to check if set explicitly by user except
  // indirectly via setFunctionAttributes()...
  return codegen::getFramePointerUsage();
#else
  if (::FramePointerUsage.getNumOccurrences() > 0)
    return ::FramePointerUsage.getValue();
  if (disableFPElim.getNumOccurrences() > 0)
    return disableFPElim ? llvm::FramePointer::All : llvm::FramePointer::None;
  return std::nullopt;
#endif
}

bool disableRedZone() { return ::DisableRedZone; }

bool printTargetFeaturesHelp() {
#if LDC_LLVM_VER >= 1100
  const auto MCPU = codegen::getMCPU();
  const auto MAttrs = codegen::getMAttrs();
#endif
  if (MCPU == "help")
    return true;
  return std::any_of(MAttrs.begin(), MAttrs.end(),
                     [](const std::string &a) { return a == "help"; });
}

TargetOptions InitTargetOptionsFromCodeGenFlags(const llvm::Triple &triple) {
#if LDC_LLVM_VER >= 1200
  return codegen::InitTargetOptionsFromCodeGenFlags(triple);
#elif LDC_LLVM_VER >= 1100
  return codegen::InitTargetOptionsFromCodeGenFlags();
#else
  return ::InitTargetOptionsFromCodeGenFlags();
#endif
}

std::string getCPUStr() {
#if LDC_LLVM_VER >= 1100
  return codegen::getCPUStr();
#else
  return ::getCPUStr();
#endif
}

std::string getFeaturesStr() {
#if LDC_LLVM_VER >= 1100
  return codegen::getFeaturesStr();
#else
  return ::getFeaturesStr();
#endif
}

#if LDC_LLVM_VER >= 1000
void setFunctionAttributes(StringRef cpu, StringRef features,
                           Function &function) {
#if LDC_LLVM_VER >= 1100
  return codegen::setFunctionAttributes(cpu, features, function);
#else
  return ::setFunctionAttributes(cpu, features, function);
#endif
}
#endif
} // namespace opts

#if LDC_WITH_LLD
// LLD uses the shared header too (for LTO) and exposes some wrappers in
// the lld namespace. Define them here to prevent the LLD object from being
// linked in with its conflicting command-line options.
namespace lld {
TargetOptions initTargetOptionsFromCodeGenFlags() {
  return ::opts::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
}

#if LDC_LLVM_VER >= 1000
Optional<Reloc::Model> getRelocModelFromCMModel() {
  return ::opts::getRelocModel();
}
#endif

Optional<CodeModel::Model> getCodeModelFromCMModel() {
  return ::opts::getCodeModel();
}

std::string getCPUStr() { return ::opts::getCPUStr(); }

#if LDC_LLVM_VER >= 1100
std::vector<std::string> getMAttrs() { return codegen::getMAttrs(); }
#else
std::vector<std::string> getMAttrs() { return ::MAttrs; }
#endif
} // namespace lld
#endif // LDC_WITH_LLD
