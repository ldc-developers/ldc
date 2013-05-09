//===-- target.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/target.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mars.h"

llvm::TargetMachine* createTargetMachine(
    std::string targetTriple,
    std::string arch,
    std::string cpu,
    std::vector<std::string> attrs,
    ExplicitBitness::Type bitness,
    llvm::Reloc::Model relocModel,
    llvm::CodeModel::Model codeModel,
    llvm::CodeGenOpt::Level codeGenOptLevel,
    bool genDebugInfo)
{
    // override triple if needed
    std::string defaultTriple = llvm::sys::getDefaultTargetTriple();
    if (sizeof(void*) == 4 && bitness == ExplicitBitness::M64)
    {
#if LDC_LLVM_VER >= 301
        defaultTriple = llvm::Triple(defaultTriple).get64BitArchVariant().str();
#else
        defaultTriple = llvm::Triple__get64BitArchVariant(defaultTriple).str();
#endif
    }
    else if (sizeof(void*) == 8 && bitness == ExplicitBitness::M32)
    {
#if LDC_LLVM_VER >= 301
        defaultTriple = llvm::Triple(defaultTriple).get32BitArchVariant().str();
#else
        defaultTriple = llvm::Triple__get32BitArchVariant(defaultTriple).str();
#endif
    }

    std::string triple;

    // did the user override the target triple?
    if (targetTriple.empty())
    {
        if (!arch.empty())
        {
            error("you must specify a target triple as well with -mtriple when using the -arch option");
            fatal();
        }
        triple = defaultTriple;
    }
    else
    {
        triple = llvm::Triple::normalize(targetTriple);
    }


    // Allocate target machine.
    const llvm::Target *theTarget = NULL;
    // Check whether the user has explicitly specified an architecture to compile for.
    if (arch.empty())
    {
        std::string Err;
        theTarget = llvm::TargetRegistry::lookupTarget(triple, Err);
        if (theTarget == 0)
        {
            error("%s Please use the -march option.", Err.c_str());
            fatal();
        }
    }
    else
    {
        for (llvm::TargetRegistry::iterator it = llvm::TargetRegistry::begin(),
             ie = llvm::TargetRegistry::end(); it != ie; ++it)
        {
            if (arch == it->getName())
            {
                theTarget = &*it;
                break;
            }
        }

        if (!theTarget)
        {
            error("invalid target '%s'", arch.c_str());
            fatal();
        }
    }

    // Package up features to be passed to target/subtarget
    std::string FeaturesStr;
    if (cpu.size() || attrs.size())
    {
        llvm::SubtargetFeatures Features;
        for (unsigned i = 0; i != attrs.size(); ++i)
            Features.AddFeature(attrs[i]);
        FeaturesStr = Features.getString();
    }

#if LDC_LLVM_VER == 300
    llvm::NoFramePointerElim = genDebugInfo;

    return theTarget->createTargetMachine(triple, cpu, FeaturesStr,
        relocModel, codeModel);
#else
    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = genDebugInfo;

    return theTarget->createTargetMachine(
        triple,
        cpu,
        FeaturesStr,
        targetOptions,
        relocModel,
        codeModel,
        codeGenOptLevel
    );
#endif
}
