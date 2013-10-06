//===-- targetmachine.cpp -------------------------------------------------===//
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

#include "driver/targetmachine.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mars.h"
#include "gen/logger.h"


static std::string getX86TargetCPU(const llvm::Triple &triple)
{
    // Select the default CPU if none was given (or detection failed).

    // Intel Macs are relatively recent, take advantage of that.
    if (triple.isOSDarwin())
        return triple.isArch64Bit() ? "core2" : "yonah";

    // Everything else goes to x86-64 in 64-bit mode.
    if (triple.isArch64Bit())
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

static std::string getARMTargetCPU(const llvm::Triple &triple)
{
    const char *result = llvm::StringSwitch<const char *>(triple.getArchName())
        .Cases("armv2", "armv2a","arm2")
        .Case("armv3", "arm6")
        .Case("armv3m", "arm7m")
        .Case("armv4", "strongarm")
        .Case("armv4t", "arm7tdmi")
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
        .Cases("armv8", "armv8a", "armv8-a", "cortex-a53")
        .Case("ep9312", "ep9312")
        .Case("iwmmxt", "iwmmxt")
        .Case("xscale", "xscale")
        // If all else failed, return the most base CPU with thumb interworking
        // supported by LLVM.
        .Default(0);

    if (result)
        return result;

    return (triple.getEnvironment() == llvm::Triple::GNUEABIHF) ?
        "arm1176jzf-s" : "arm7tdmi";
}

/// Returns the LLVM name of the target CPU to use given the provided
/// -mcpu argument and target triple.
static std::string getTargetCPU(const std::string &cpu,
    const llvm::Triple &triple)
{
    if (!cpu.empty())
    {
        if (cpu != "native")
            return cpu;

        // FIXME: Reject attempts to use -mcpu=native unless the target matches
        // the host.
        std::string hostCPU = llvm::sys::getHostCPUName();
        if (!hostCPU.empty() && hostCPU != "generic")
            return hostCPU;
    }

    if (triple.getArch() == llvm::Triple::x86_64 ||
        triple.getArch() == llvm::Triple::x86)
    {
        return getX86TargetCPU(triple);
    }
    else if (triple.getArch() == llvm::Triple::arm)
    {
        return getARMTargetCPU(triple);
    }

    return cpu;
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
    // Determine target triple. If the user didn't explicitly specify one, use
    // the one set at LLVM configure time.
    llvm::Triple triple;
    if (targetTriple.empty())
    {
        triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());

        // Handle -m32/-m64.
        if (sizeof(void*) == 4 && bitness == ExplicitBitness::M64)
        {
            triple = triple.get64BitArchVariant();
        }
        else if (sizeof(void*) == 8 && bitness == ExplicitBitness::M32)
        {
            triple = triple.get32BitArchVariant();
        }
    }
    else
    {
        triple = llvm::Triple(llvm::Triple::normalize(targetTriple));
    }

    // Look up the LLVM backend to use. This also updates triple with the
    // user-specified arch, if any.
    std::string errMsg;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(arch, triple, errMsg);
    if (target == 0)
    {
        error("Could not determine target platform: %s", errMsg.c_str());
        fatal();
    }

    // Package up features to be passed to target/subtarget.
    llvm::SubtargetFeatures features;
    features.getDefaultSubtargetFeatures(triple);
    if (cpu == "native")
    {
        llvm::StringMap<bool> hostFeatures;
        if (llvm::sys::getHostCPUFeatures(hostFeatures))
        {
            llvm::StringMapConstIterator<bool> i = hostFeatures.begin(),
                end = hostFeatures.end();
            for (; i != end; ++i)
                features.AddFeature(i->first(), i->second);
        }
    }
    for (unsigned i = 0; i < attrs.size(); ++i)
        features.AddFeature(attrs[i]);

    // With an empty CPU string, LLVM will default to the host CPU, which is
    // usually not what we want (expected behavior from other compilers is
    // to default to "generic").
    cpu = getTargetCPU(cpu, triple);

    if (Logger::enabled())
    {
        Logger::cout() << "Targeting '" << triple.str() << "' (CPU '" << cpu
            << "' with features '" << features.getString() << "').\n";
    }

    if (triple.isMacOSX() && relocModel == llvm::Reloc::Default)
    {
        // OS X defaults to PIC (and as of 10.7.5/LLVM 3.1-3.3, TLS use leads
        // to crashes for non-PIC code). LLVM doesn't handle this.
        relocModel = llvm::Reloc::PIC_;
    }

    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = genDebugInfo;

    return target->createTargetMachine(
        triple.str(),
        cpu,
        features.getString(),
        targetOptions,
        relocModel,
        codeModel,
        codeGenOptLevel
    );
}
