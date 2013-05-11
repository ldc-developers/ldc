//===-- target.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Note: The target CPU and FP ABI detection logic has been adapted from Clang
// (Tools.cpp and ToolChain.cpp in lib/Driver, the latter seems to have the
// more up-to-date version).
//
//===----------------------------------------------------------------------===//

#include "driver/target.h"
#include "gen/llvmcompat.h"
#include "llvm/ADT/StringSwitch.h"
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


static const char *getLLVMArchSuffixForARM(llvm::StringRef CPU)
{
    return llvm::StringSwitch<const char *>(CPU)
        .Cases("arm7tdmi", "arm7tdmi-s", "arm710t", "v4t")
        .Cases("arm720t", "arm9", "arm9tdmi", "v4t")
        .Cases("arm920", "arm920t", "arm922t", "v4t")
        .Cases("arm940t", "ep9312","v4t")
        .Cases("arm10tdmi",  "arm1020t", "v5")
        .Cases("arm9e",  "arm926ej-s",  "arm946e-s", "v5e")
        .Cases("arm966e-s",  "arm968e-s",  "arm10e", "v5e")
        .Cases("arm1020e",  "arm1022e",  "xscale", "iwmmxt", "v5e")
        .Cases("arm1136j-s",  "arm1136jf-s",  "arm1176jz-s", "v6")
        .Cases("arm1176jzf-s",  "mpcorenovfp",  "mpcore", "v6")
        .Cases("arm1156t2-s",  "arm1156t2f-s", "v6t2")
        .Cases("cortex-a5", "cortex-a7", "cortex-a8", "v7")
        .Cases("cortex-a9", "cortex-a15", "v7")
        .Case("cortex-r5", "v7r")
        .Case("cortex-m0", "v6m")
        .Case("cortex-m3", "v7m")
        .Case("cortex-m4", "v7em")
        .Case("cortex-a9-mp", "v7f")
        .Case("swift", "v7s")
        .Default("");
}

static std::string getARMTargetCPU(std::string arch, const llvm::Triple &triple)
{
    // FIXME: Warn on inconsistent use of -mcpu and -march.

    llvm::StringRef MArch;
    if (!arch.empty()) {
        // Otherwise, if we have -march= choose the base CPU for that arch.
        MArch = arch;
    } else {
        // Otherwise, use the Arch from the triple.
        MArch = triple.getArchName();
    }

    // Handle -march=native.
    std::string NativeMArch;
    if (MArch == "native") {
        std::string CPU = llvm::sys::getHostCPUName();
        if (CPU != "generic") {
            // Translate the native cpu into the architecture. The switch below will
            // then chose the minimum cpu for that arch.
            NativeMArch = std::string("arm") + getLLVMArchSuffixForARM(CPU);
            MArch = NativeMArch;
        }
    }

    return llvm::StringSwitch<const char *>(MArch)
        .Cases("armv2", "armv2a","arm2")
        .Case("armv3", "arm6")
        .Case("armv3m", "arm7m")
        .Cases("armv4", "armv4t", "arm7tdmi")
        .Cases("armv5", "armv5t", "arm10tdmi")
        .Cases("armv5e", "armv5te", "arm1026ejs")
        .Case("armv5tej", "arm926ej-s")
        .Cases("armv6", "armv6k", "arm1136jf-s")
        .Case("armv6j", "arm1136j-s")
        .Cases("armv6z", "armv6zk", "arm1176jzf-s")
        .Case("armv6t2", "arm1156t2-s")
        .Cases("armv6m", "armv6-m", "cortex-m0")
        .Cases("armv7", "armv7a", "armv7-a", "cortex-a8")
        .Cases("armv7l", "armv7-l", "cortex-a8")
        .Cases("armv7f", "armv7-f", "cortex-a9-mp")
        .Cases("armv7s", "armv7-s", "swift")
        .Cases("armv7r", "armv7-r", "cortex-r4")
        .Cases("armv7m", "armv7-m", "cortex-m3")
        .Cases("armv7em", "armv7e-m", "cortex-m4")
        .Case("ep9312", "ep9312")
        .Case("iwmmxt", "iwmmxt")
        .Case("xscale", "xscale")
        // If all else failed, return the most base CPU LLVM supports.
        .Default("arm7tdmi");
}

static FloatABI::Type getARMFloatABI(const llvm::Triple &triple,
    const char* llvmArchSuffix)
{
    switch (triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS: {
        // Darwin defaults to "softfp" for v6 and v7.
        if (llvm::StringRef(llvmArchSuffix).startswith("v6") ||
            llvm::StringRef(llvmArchSuffix).startswith("v7"))
            return FloatABI::SoftFP;
        return FloatABI::Soft;
    }

    case llvm::Triple::FreeBSD:
        // FreeBSD defaults to soft float
        return FloatABI::Soft;

    default:
        switch(triple.getEnvironment()) {
        case llvm::Triple::GNUEABIHF:
            return FloatABI::Hard;
        case llvm::Triple::GNUEABI:
            return FloatABI::SoftFP;
        case llvm::Triple::EABI:
            // EABI is always AAPCS, and if it was not marked 'hard', it's softfp
            return FloatABI::SoftFP;
        case llvm::Triple::Android: {
            if (llvm::StringRef(llvmArchSuffix).startswith("v7"))
                return FloatABI::SoftFP;
            return FloatABI::Soft;
        }
        default:
            // Assume "soft".
            // TODO: Warn the user we are guessing.
            return FloatABI::Soft;
        }
    }
}


llvm::TargetMachine* createTargetMachine(
    std::string targetTriple,
    std::string arch,
    std::string cpu,
    std::vector<std::string> attrs,
    ExplicitBitness::Type bitness,
    FloatABI::Type floatABI,
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
        switch (triple.getArch())
        {
        default: break;
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            cpu = getX86TargetCPU(arch, triple);
            break;
        case llvm::Triple::arm:
            cpu = getARMTargetCPU(arch, triple);
            break;
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

    if (floatABI == FloatABI::Default)
    {
        switch (triple.getArch())
        {
        default: // X86, ...
            floatABI = FloatABI::Hard;
            break;
        case llvm::Triple::arm:
            floatABI = getARMFloatABI(triple, getLLVMArchSuffixForARM(cpu));
            break;
        case llvm::Triple::thumb:
            floatABI = FloatABI::Soft;
            break;
        }
    }

#if LDC_LLVM_VER == 300
    llvm::NoFramePointerElim = genDebugInfo;
    // FIXME: Handle floating-point ABI.

    return theTarget->createTargetMachine(triple.str(), cpu, FeaturesStr,
        relocModel, codeModel);
#else
    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = genDebugInfo;

    switch (floatABI)
    {
    default: llvm_unreachable("Floating point ABI type unknown.");
    case FloatABI::Soft:
        targetOptions.UseSoftFloat = true;
        targetOptions.FloatABIType = llvm::FloatABI::Soft;
        break;
    case FloatABI::SoftFP:
        targetOptions.UseSoftFloat = false;
        targetOptions.FloatABIType = llvm::FloatABI::Soft;
        break;
    case FloatABI::Hard:
        targetOptions.UseSoftFloat = false;
        targetOptions.FloatABIType = llvm::FloatABI::Hard;
        break;
    }

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
