//===-- cl_options-llvm.cpp -----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options-llvm.h"

// Pull in command-line options and helper functions from special LLVM header
// shared by multiple LLVM tools.
#if LDC_LLVM_VER >= 700
#include "llvm/CodeGen/CommandFlags.inc"
#elif LDC_LLVM_VER >= 600
#include "llvm/CodeGen/CommandFlags.def"
#else
#include "llvm/CodeGen/CommandFlags.h"
#endif

static cl::opt<bool>
    DisableRedZone("disable-red-zone", cl::ZeroOrMore,
                   cl::desc("Do not emit code that uses the red zone."));

// Now expose the helper functions (with static linkage) via external wrappers
// in the opts namespace, including some additional helper functions.
namespace opts {

std::string getArchStr() { return ::MArch; }

Optional<Reloc::Model> getRelocModel() { return ::getRelocModel(); }

#if LDC_LLVM_VER >= 600
Optional<CodeModel::Model> getCodeModel() { return ::getCodeModel(); }
#else
CodeModel::Model getCodeModel() { return ::CMModel; }
#endif

#if LDC_LLVM_VER >= 900
llvm::Optional<llvm::FramePointer::FP> framePointerUsage() {
  if (::FramePointerUsage.getNumOccurrences() == 0)
    return llvm::None;
  else
    return ::FramePointerUsage.getValue();
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
  if (MCPU == "help")
    return true;
  return std::any_of(MAttrs.begin(), MAttrs.end(),
                     [](const std::string &a) { return a == "help"; });
}

TargetOptions InitTargetOptionsFromCodeGenFlags() {
  return ::InitTargetOptionsFromCodeGenFlags();
}

std::string getCPUStr() { return ::getCPUStr(); }
std::string getFeaturesStr() { return ::getFeaturesStr(); }
} // namespace opts

#if LDC_WITH_LLD && LDC_LLVM_VER >= 500
// LLD 5.0 uses the shared header too (for LTO) and exposes some wrappers in
// the lld namespace. Define them here to prevent the LLD object from being
// linked in with its conflicting command-line options.
namespace lld {
TargetOptions InitTargetOptionsFromCodeGenFlags() {
  return ::InitTargetOptionsFromCodeGenFlags();
}

#if LDC_LLVM_VER >= 600
Optional<CodeModel::Model> GetCodeModelFromCMModel() {
  return ::getCodeModel();
}
#else
CodeModel::Model GetCodeModelFromCMModel() { return ::CMModel; }
#endif

#if LDC_LLVM_VER >= 700
std::string GetCPUStr() { return ::getCPUStr(); }
#endif
}
#endif // LDC_WITH_LLD && LDC_LLVM_VER >= 500
