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
#if LDC_LLVM_VER <= 306
#include "llvm/Support/Host.h"
#include "llvm/MC/SubtargetFeature.h"
#endif

#if LDC_LLVM_VER > 305
static cl::opt<bool>
    DisableRedZone("disable-red-zone", cl::ZeroOrMore,
                   cl::desc("Do not emit code that uses the red zone."));
#endif

#if LDC_LLVM_VER >= 800
// legacy option
static cl::opt<bool>
    disableFPElim("disable-fp-elim", cl::ZeroOrMore, cl::ReallyHidden,
                  cl::desc("Disable frame pointer elimination optimization"));
#endif

// Now expose the helper functions (with static linkage) via external wrappers
// in the opts namespace, including some additional helper functions.
namespace opts {

std::string getArchStr() { return ::MArch; }

#if LDC_LLVM_VER >= 309
Optional<Reloc::Model> getRelocModel() { return ::getRelocModel(); }
#else
Reloc::Model getRelocModel() { return ::RelocModel; }
#endif

#if LDC_LLVM_VER >= 600
Optional<CodeModel::Model> getCodeModel() { return ::getCodeModel(); }
#else
CodeModel::Model getCodeModel() { return ::CMModel; }
#endif

#if LDC_LLVM_VER >= 800
llvm::Optional<llvm::FramePointer::FP> framePointerUsage() {
  if (::FramePointerUsage.getNumOccurrences() > 0)
    return ::FramePointerUsage.getValue();
  if (disableFPElim.getNumOccurrences() > 0)
    return disableFPElim ? llvm::FramePointer::All : llvm::FramePointer::None;
  return llvm::None;
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

std::string getCPUStr() {
#if LDC_LLVM_VER < 307
  if (::MCPU == "native")
    return ::sys::getHostCPUName();

  return ::MCPU;
#else
  return ::getCPUStr();
#endif
}

std::string getFeaturesStr() {
#if LDC_LLVM_VER < 307
  SubtargetFeatures Features;

  // If user asked for the 'native' CPU, we need to autodetect features.
  // This is necessary for x86 where the CPU might not support all the
  // features the autodetected CPU name lists in the target. For example,
  // not all Sandybridge processors support AVX.
  if (MCPU == "native") {
    StringMap<bool> HostFeatures;
    if (sys::getHostCPUFeatures(HostFeatures))
      for (auto &F : HostFeatures)
        Features.AddFeature(std::string(F.second ? "+" : "-").append(F.first()));
  }

  for (unsigned i = 0; i != MAttrs.size(); ++i)
    Features.AddFeature(MAttrs[i]);

  return Features.getString();
#else
  return ::getFeaturesStr();
#endif
}
} // namespace opts

#if LDC_WITH_LLD && LDC_LLVM_VER >= 500
// LLD 5.0 uses the shared header too (for LTO) and exposes some wrappers in
// the lld namespace. Define them here to prevent the LLD object from being
// linked in with its conflicting command-line options.
namespace lld {
TargetOptions InitTargetOptionsFromCodeGenFlags() {
  return ::InitTargetOptionsFromCodeGenFlags();
}

CodeModel::Model GetCodeModelFromCMModel() { return CMModel; }
}
#endif // LDC_WITH_LLD && LDC_LLVM_VER >= 500
