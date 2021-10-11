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
#elif LDC_LLVM_VER >= 700
#include "llvm/CodeGen/CommandFlags.inc"
#else
#include "llvm/CodeGen/CommandFlags.def"
#endif

static cl::opt<bool>
    DisableRedZone("disable-red-zone", cl::ZeroOrMore,
                   cl::desc("Do not emit code that uses the red zone."));

#if LDC_LLVM_VER >= 800 && LDC_LLVM_VER < 1100
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
#elif LDC_LLVM_VER >= 800
using FPK = llvm::FramePointer::FP;
#endif

#if LDC_LLVM_VER >= 800
llvm::Optional<FPK> framePointerUsage() {
#if LDC_LLVM_VER >= 1100
  // Defaults to `FP::None`; no way to check if set explicitly by user except
  // indirectly via setFunctionAttributes()...
  return codegen::getFramePointerUsage();
#else
  if (::FramePointerUsage.getNumOccurrences() > 0)
    return ::FramePointerUsage.getValue();
  if (disableFPElim.getNumOccurrences() > 0)
    return disableFPElim ? llvm::FramePointer::All : llvm::FramePointer::None;
  return llvm::None;
#endif
}
#else
cl::boolOrDefault disableFPElim() {
  return ::DisableFPElim.getNumOccurrences() == 0
             ? cl::BOU_UNSET
             : ::DisableFPElim ? cl::BOU_TRUE : cl::BOU_FALSE;
}
#endif

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
#if LDC_LLVM_VER >= 900
TargetOptions initTargetOptionsFromCodeGenFlags() {
#else
TargetOptions InitTargetOptionsFromCodeGenFlags() {
#endif
  return ::opts::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
}

#if LDC_LLVM_VER >= 1000
Optional<Reloc::Model> getRelocModelFromCMModel() {
  return ::opts::getRelocModel();
}
#endif

#if LDC_LLVM_VER >= 900
Optional<CodeModel::Model> getCodeModelFromCMModel() {
#else
Optional<CodeModel::Model> GetCodeModelFromCMModel() {
#endif
  return ::opts::getCodeModel();
}

#if LDC_LLVM_VER >= 900
std::string getCPUStr() { return ::opts::getCPUStr(); }
#elif LDC_LLVM_VER >= 700
std::string GetCPUStr() { return ::opts::getCPUStr(); }
#endif

#if LDC_LLVM_VER >= 1100
std::vector<std::string> getMAttrs() { return codegen::getMAttrs(); }
#elif LDC_LLVM_VER >= 900
std::vector<std::string> getMAttrs() { return ::MAttrs; }
#elif LDC_LLVM_VER >= 800
std::vector<std::string> GetMAttrs() { return ::MAttrs; }
#endif
} // namespace lld
#endif // LDC_WITH_LLD
