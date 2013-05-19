//===-- target.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Note: The target CPU detection logic has been adapted from Clang
// (lib/Driver/Tools.cpp).
//
//===----------------------------------------------------------------------===//

#include "driver/target.h"
#include "gen/llvmcompat.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mars.h"


static std::string getX86TargetCPU(std::string arch,
    const llvm::Triple &triple)
{
    if (!arch.empty()) {
        if (arch != "native")
          return arch;

        // FIXME: Reject attempts to use -march=native unless the target matches
        // the host.
        //
        // FIXME: We should also incorporate the detected target features for use
        // with -native.
        std::string cpu = llvm::sys::getHostCPUName();
        if (!cpu.empty() && cpu != "generic")
          return cpu;
    }

    // Select the default CPU if none was given (or detection failed).

    bool is64Bit = triple.getArch() == llvm::Triple::x86_64;

    if (triple.isOSDarwin())
        return is64Bit ? "core2" : "yonah";

    // Everything else goes to x86-64 in 64-bit mode.
    if (is64Bit)
        return "x86-64";

    if (triple.getOSName().startswith("haiku"))
        return "i586";
    if (triple.getOSName().startswith("openbsd"))
        return "i486";
    if (triple.getOSName().startswith("bitrig"))
        return "i686";
    if (triple.getOSName().startswith("freebsd"))
        return "i486";
    if (triple.getOSName().startswith("netbsd"))
        return "i486";
#if LDC_LLVM_VER >= 302
    // All x86 devices running Android have core2 as their common
    // denominator. This makes a better choice than pentium4.
    if (triple.getEnvironment() == llvm::Triple::Android)
        return "core2";
#endif

    // Fallback to p4.
    return "pentium4";
}

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

    llvm::Triple triple;

    // did the user override the target triple?
    if (targetTriple.empty())
    {
        if (!arch.empty())
        {
            error("you must specify a target triple as well with -mtriple when using the -arch option");
            fatal();
        }
        triple = llvm::Triple(defaultTriple);
    }
    else
    {
        triple = llvm::Triple(llvm::Triple::normalize(targetTriple));
    }

    // Allocate target machine.
    const llvm::Target *theTarget = NULL;
    // Check whether the user has explicitly specified an architecture to compile for.
    if (arch.empty())
    {
        std::string Err;
        theTarget = llvm::TargetRegistry::lookupTarget(triple.str(), Err);
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


    // With an empty CPU string, LLVM will default to the host CPU, which is
    // usually not what we want (expected behavior from other compilers is
    // to default to "generic").
    if (cpu.empty())
    {
        if (triple.getArch() == llvm::Triple::x86_64 ||
            triple.getArch() == llvm::Triple::x86)
        {
            cpu = getX86TargetCPU(arch, triple);
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

    if (triple.isMacOSX() && relocModel == llvm::Reloc::Default)
    {
        // OS X defaults to PIC (and as of 10.7.5/LLVM 3.1-3.3, TLS use leads
        // to crashes for non-PIC code). LLVM doesn't handle this.
        relocModel = llvm::Reloc::PIC_;
    }

#if LDC_LLVM_VER == 300
    llvm::NoFramePointerElim = genDebugInfo;

    return theTarget->createTargetMachine(triple.str(), cpu, FeaturesStr,
        relocModel, codeModel);
#else
    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = genDebugInfo;

    return theTarget->createTargetMachine(
        triple.str(),
        cpu,
        FeaturesStr,
        targetOptions,
        relocModel,
        codeModel,
        codeGenOptLevel
    );
#endif
}
