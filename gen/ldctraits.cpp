//===-- ldctraits.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/irstate.h"
#include "gen/ldctraits.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCSubtargetInfo.h"

Dstring traitsGetTargetCPU() {
  auto cpu = gTargetMachine->getTargetCPU();
  return {cpu.size(), cpu.data()};
}

bool traitsTargetHasFeature(Dstring feature) {
  auto feat = llvm::StringRef(feature.ptr, feature.length);

  // This is a work-around to a missing interface in LLVM to query whether a
  // feature is set.

  // Copy MCSubtargetInfo so we can modify it.
  llvm::MCSubtargetInfo mcinfo = *gTargetMachine->getMCSubtargetInfo();
  auto savedFeatbits = mcinfo.getFeatureBits();

  // Nothing will change if the feature string is not recognized or if the
  // feature is disabled.
  {
    auto newFeatbits = mcinfo.ApplyFeatureFlag(("-" + feat).str());
    if (savedFeatbits == newFeatbits) {
      return false;
    }
    mcinfo.setFeatureBits(savedFeatbits);
  }
  {
    // Now that unrecognized feature strings are excluded,
    // nothing will change iff the feature and its implied features are enabled.
    auto newFeatbits = mcinfo.ApplyFeatureFlag(("+" + feat).str());
    return savedFeatbits == newFeatbits;
  }
}
