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
// (Tools.cpp and ToolChain.cpp in lib/Driver, the latter seems to have the
// more up-to-date version).
//
//===----------------------------------------------------------------------===//

#include "driver/targetmachine.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/CommandLine.h"
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

    switch (triple.getArch())
    {
    default:
        // We don't know about the specifics of this platform, just return the
        // empty string and let LLVM decide.
        return cpu;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        return getX86TargetCPU(triple);
    case llvm::Triple::arm:
        return getARMTargetCPU(triple);
    }
}

static const char *getLLVMArchSuffixForARM(llvm::StringRef CPU)
{
    return llvm::StringSwitch<const char *>(CPU)
        .Case("strongarm", "v4")
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
        .Cases("cortex-a9", "cortex-a12", "cortex-a15", "v7")
        .Cases("cortex-r4", "cortex-r5", "v7r")
        .Case("cortex-m0", "v6m")
        .Case("cortex-m3", "v7m")
        .Case("cortex-m4", "v7em")
        .Case("cortex-a9-mp", "v7f")
        .Case("swift", "v7s")
        .Case("cortex-a53", "v8")
        .Case("krait", "v7")
        .Default("");
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
#if LDC_LLVM_VER >= 302
        case llvm::Triple::Android: {
            if (llvm::StringRef(llvmArchSuffix).startswith("v7"))
                return FloatABI::SoftFP;
            return FloatABI::Soft;
        }
#endif
        default:
            // Assume "soft".
            // TODO: Warn the user we are guessing.
            return FloatABI::Soft;
        }
    }
}

/// Sanitizes the MIPS ABI in the feature string.
static void addMipsABI(const llvm::Triple &triple, std::vector<std::string> &attrs)
{
    enum ABI { O32 = 1<<0, N32 = 1<<1, N64 = 1<<2, EABI = 1<<3 };
    const bool is64Bit = triple.getArch() == llvm::Triple::mips64 ||
                         triple.getArch() == llvm::Triple::mips64el;
    const uint32_t defaultABI = is64Bit ? N64 : O32;
    uint32_t bits = defaultABI;
    std::vector<std::string>::iterator I = attrs.begin();
    while (I != attrs.end())
    {
        std::string str = *I;
        bool enabled = str[0] == '+';
        std::string flag = (str[0] == '+' || str[0] == '-') ? str.substr(1) : str;
        uint32_t newBit = 0;
        if (flag == "o32") newBit = O32;
        if (flag == "n32") newBit = N32;
        if (flag == "n64") newBit = N64;
        if (flag == "eabi") newBit = EABI;
        if (newBit)
        {
            I = attrs.erase(I);
            if (enabled) bits |= newBit;
            else bits &= ~newBit;
        }
        else
            ++I;
    }
    switch (bits)
    {
        case O32: attrs.push_back("+o32"); break;
        case N32: attrs.push_back("+n32"); break;
        case N64: attrs.push_back("+n64"); break;
        case EABI: attrs.push_back("+eabi"); break;
        default: error(Loc(), "Only one ABI argument is supported"); fatal();
    }
    if (bits != defaultABI)
        attrs.push_back(is64Bit ? "-n64" : "-o32");
}

/// Looks up a target based on an arch name and a target triple.
///
/// If the arch name is non-empty, then the lookup is done by arch. Otherwise,
/// the target triple is used.
///
/// This has been adapted from the corresponding LLVM 3.2+ overload of
/// llvm::TargetRegistry::lookupTarget. Once support for LLVM 3.1 is dropped,
/// the registry method can be used instead.
const llvm::Target *lookupTarget(const std::string &arch, llvm::Triple &triple,
    std::string &errorMsg)
{
    // Allocate target machine. First, check whether the user has explicitly
    // specified an architecture to compile for. If so we have to look it up by
    // name, because it might be a backend that has no mapping to a target triple.
    const llvm::Target *target = 0;
    if (!arch.empty())
    {
        for (llvm::TargetRegistry::iterator it = llvm::TargetRegistry::begin(),
            ie = llvm::TargetRegistry::end(); it != ie; ++it)
        {
            if (arch == it->getName())
            {
                target = &*it;
                break;
            }
        }

        if (!target)
        {
            errorMsg = "invalid target architecture '" + arch + "', see "
                "-version for a list of supported targets.";
            return 0;
        }

        // Adjust the triple to match (if known), otherwise stick with the
        // given triple.
        llvm::Triple::ArchType Type = llvm::Triple::getArchTypeForLLVMName(arch);
        if (Type != llvm::Triple::UnknownArch)
            triple.setArch(Type);
    }
    else
    {
        std::string tempError;
        target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), tempError);
        if (!target)
        {
            errorMsg = "unable to get target for '" + triple.getTriple() +
                "', see -version and -mtriple.";
        }
    }

    return target;
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
    bool noFramePointerElim,
    bool noLinkerStripDead,
    bool enableFPMAD,
    bool enableUnsafeFPMath,
    bool enableNoInfsFPMath,
    bool enableNoNaNsFPMath,
    bool enableHonorSignDependentRoundingFPMath)
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
    const llvm::Target *target = lookupTarget(arch, triple, errMsg);
    if (target == 0)
    {
        error(Loc(), "%s", errMsg.c_str());
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
#if LDC_LLVM_VER >= 305
                features.AddFeature(std::string((i->second ? "+" : "-")).append(i->first()));
#else
                features.AddFeature(i->first(), i->second);
#endif
        }
    }
    if (triple.getArch() == llvm::Triple::mips ||
        triple.getArch() == llvm::Triple::mipsel ||
        triple.getArch() == llvm::Triple::mips64 ||
        triple.getArch() == llvm::Triple::mips64el)
        addMipsABI(triple, attrs);
    for (unsigned i = 0; i < attrs.size(); ++i)
        features.AddFeature(attrs[i]);

    // With an empty CPU string, LLVM will default to the host CPU, which is
    // usually not what we want (expected behavior from other compilers is
    // to default to "generic").
    cpu = getTargetCPU(cpu, triple);

    if (Logger::enabled())
    {
        Logger::println("Targeting '%s' (CPU '%s' with features '%s')",
            triple.str().c_str(), cpu.c_str(), features.getString().c_str());
    }

    if (triple.isMacOSX() && relocModel == llvm::Reloc::Default)
    {
        // OS X defaults to PIC (and as of 10.7.5/LLVM 3.1-3.3, TLS use leads
        // to crashes for non-PIC code). LLVM doesn't handle this.
        relocModel = llvm::Reloc::PIC_;
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

#if LDC_LLVM_VER < 305
    if (triple.getArch() == llvm::Triple::arm && !triple.isOSDarwin())
    {
        // On ARM, we want to use EHABI exception handling, as we don't support
        // SJLJ EH in druntime. Unfortunately, it is still in a partly
        // experimental state, and the -arm-enable-ehabi-descriptors command
        // line option is not exposed via an internal API at all.
        const char *backendArgs[3] = {
            "ldc2", // Fake name, irrelevant.
            "-arm-enable-ehabi",
            "-arm-enable-ehabi-descriptors"
        };
        llvm::cl::ParseCommandLineOptions(3, backendArgs);
    }
#endif

    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = noFramePointerElim;

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

    // Initialize various math options.
    targetOptions.LessPreciseFPMADOption = enableFPMAD;
    targetOptions.UnsafeFPMath = enableUnsafeFPMath;
    targetOptions.NoInfsFPMath = enableNoInfsFPMath;
    targetOptions.NoNaNsFPMath = enableNoNaNsFPMath;
    targetOptions.HonorSignDependentRoundingFPMathOption = enableHonorSignDependentRoundingFPMath;


    // Right now, we only support linker-level dead code elimination on Linux
    // using the GNU toolchain (based on ld's --gc-sections flag). The Apple ld
    // on OS X supports a similar flag (-dead_strip) that doesn't require
    // emitting the symbols into different sections. The MinGW ld doesn't seem
    // to support --gc-sections at all, and FreeBSD needs more investigation.
    if (!noLinkerStripDead &&
        (triple.getOS() == llvm::Triple::Linux || triple.getOS() == llvm::Triple::Win32))
    {
#if LDC_LLVM_VER < 305
        llvm::TargetMachine::setDataSections(true);
        llvm::TargetMachine::setFunctionSections(true);
#else
        targetOptions.FunctionSections = true;
        targetOptions.DataSections = true;
#endif
    }

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
