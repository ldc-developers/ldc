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

#include "dmd/errors.h"
#include "driver/args.h"
#include "driver/cl_options.h"
#include "gen/logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetParser.h"
#if LDC_LLVM_VER >= 1400
#include "llvm/Support/AArch64TargetParser.h"
#include "llvm/Support/ARMTargetParser.h"
#endif
#else
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#endif
#include "llvm/IR/Module.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/CommandLine.h"
#if LDC_LLVM_VER >= 1400
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gen/optimizer.h"

#ifdef LDC_LLVM_SUPPORTS_MACHO_DWARF_LINE_AS_REGULAR_SECTION
// LDC-LLVM >= 6.0.1:
// On Mac, emit __debug_line section in __DWARF segment as regular (non-debug)
// section, like DMD, to enable file/line infos in backtraces. See
// https://github.com/dlang/dmd/commit/2bf7d0db29416eacbb01a91e6502140e354ee0ef
// https://github.com/ldc-developers/llvm-project/commit/110deda1bc1cf195983fea8c1107886057987955
static llvm::cl::opt<bool, true> preserveDwarfLineSection(
    "preserve-dwarf-line-section",
    llvm::cl::desc("Mac: preserve DWARF line section during linking for "
                   "file/line infos in backtraces. Defaults to true."),
    llvm::cl::Hidden, llvm::cl::ZeroOrMore,
    llvm::cl::location(ldc::emitMachODwarfLineAsRegularSection),
    llvm::cl::init(false));
#endif

// Returns true if 'feature' is enabled and false otherwise. Handles the
// case where the feature is specified multiple times ('+m,-m'), and
// takes the last occurrence.
bool isFeatureEnabled(const llvm::SmallVectorImpl<llvm::StringRef> &features,
                      llvm::StringRef feature) {
  for (auto it = features.rbegin(), end = features.rend(); it != end; ++it) {
    if (it->substr(1) == feature) {
      return (*it)[0] == '+';
    }
  }
  return false;
};

const char *getABI(const llvm::Triple &triple, const llvm::SmallVectorImpl<llvm::StringRef> &features) {
  llvm::StringRef ABIName(opts::mABI);
  if (ABIName != "") {
    switch (triple.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
      if (ABIName.startswith("aapcs"))
        return "aapcs";
      if (ABIName.startswith("eabi"))
        return "apcs";
      break;
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
      if (ABIName.startswith("o32"))
        return "o32";
      if (ABIName.startswith("n32"))
        return "n32";
      if (ABIName.startswith("n64"))
        return "n64";
      if (ABIName.startswith("eabi"))
        return "eabi";
      break;
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
      if (ABIName.startswith("elfv1"))
        return "elfv1";
      if (ABIName.startswith("elfv2"))
        return "elfv2";
      break;
    case llvm::Triple::riscv64:
      if (ABIName.startswith("lp64f"))
        return "lp64f";
      if (ABIName.startswith("lp64d"))
        return "lp64d";
      if (ABIName.startswith("lp64"))
        return "lp64";
      break;
    case llvm::Triple::riscv32:
      if (ABIName.startswith("ilp32f"))
        return "ilp32f";
      if (ABIName.startswith("ilp32d"))
        return "ilp32d";
      if (ABIName.startswith("ilp32"))
        return "ilp32";
      break;
#if LDC_LLVM_VER >= 1600
    case llvm::Triple::loongarch32:
      if (ABIName.startswith("ilp32s"))
        return "ilp32s";
      if (ABIName.startswith("ilp32f"))
        return "ilp32f";
      if (ABIName.startswith("ilp32d"))
        return "ilp32d";
      break;
    case llvm::Triple::loongarch64:
      if (ABIName.startswith("lp64f"))
        return "lp64f";
      if (ABIName.startswith("lp64d"))
        return "lp64d";
      if (ABIName.startswith("lp64s"))
        return "lp64s";
      break;
#endif // LDC_LLVM_VER >= 1600
    default:
      break;
    }
    warning(Loc(), "Unknown ABI %s - using default ABI instead",
            ABIName.str().c_str());
  }

  switch (triple.getArch()) {
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    return "n32";
  case llvm::Triple::ppc64:
    return "elfv1";
  case llvm::Triple::ppc64le:
    return "elfv2";
  case llvm::Triple::riscv64:
    if (isFeatureEnabled(features, "d"))
      return "lp64d";
    if (isFeatureEnabled(features, "f"))
      return "lp64f";
    return "lp64";
  case llvm::Triple::riscv32:
    return "ilp32";
#if LDC_LLVM_VER >= 1600
  case llvm::Triple::loongarch32:
    if (isFeatureEnabled(features, "d"))
      return "ilp32d";
    if (isFeatureEnabled(features, "f"))
      return "ilp32f";
    return "ilp32s";
  case llvm::Triple::loongarch64:
    if (isFeatureEnabled(features, "d"))
      return "lp64d";
    if (isFeatureEnabled(features, "f"))
      return "lp64f";
    return "lp64d";
#endif // LDC_LLVM_VER >= 1600
  default:
    return "";
  }
}

extern llvm::TargetMachine *gTargetMachine;

MipsABI::Type getMipsABI() {
  // eabi can only be set on the commandline
  if (strncmp(opts::mABI.c_str(), "eabi", 4) == 0)
    return MipsABI::EABI;

  const llvm::DataLayout dl = gTargetMachine->createDataLayout();

  if (dl.getPointerSizeInBits() == 64)
    return MipsABI::N64;

  const auto largestInt = dl.getLargestLegalIntTypeSizeInBits();
  return (largestInt == 64) ? MipsABI::N32 : MipsABI::O32;
}

static std::string getX86TargetCPU(const llvm::Triple &triple) {
  // Select the default CPU if none was given (or detection failed).

  // Intel Macs are relatively recent, take advantage of that.
  if (triple.isOSDarwin()) {
    return triple.isArch64Bit() ? "core2" : "yonah";
  }

  // All x86 devices running Android have core2 as their common
  // denominator.
  if (triple.getEnvironment() == llvm::Triple::Android) {
    return "core2";
  }

  // Everything else goes to x86-64 in 64-bit mode.
  if (triple.isArch64Bit()) {
    return "x86-64";
  }
  if (triple.getOSName().startswith("haiku")) {
    return "i586";
  }
  if (triple.getOSName().startswith("openbsd")) {
    return "i486";
  }
  if (triple.getOSName().startswith("freebsd")) {
    return "i486";
  }
  if (triple.getOSName().startswith("netbsd")) {
    return "i486";
  }
  if (triple.getOSName().startswith("openbsd")) {
    return "i486";
  }
  if (triple.getOSName().startswith("dragonfly")) {
    return "i486";
  }

  // Fallback to p4.
  return "pentium4";
}

static std::string getARMTargetCPU(const llvm::Triple &triple) {
  auto defaultCPU = llvm::ARM::getDefaultCPU(triple.getArchName());

  // 32-bit Android: default to cortex-a8
  if (defaultCPU == "generic" &&
      triple.getEnvironment() == llvm::Triple::Android) {
    return "cortex-a8";
  }

  if (!defaultCPU.empty())
    return std::string(defaultCPU);

  // Return the most base CPU with thumb interworking supported by LLVM.
  return (triple.getEnvironment() == llvm::Triple::GNUEABIHF) ? "arm1176jzf-s"
                                                              : "arm7tdmi";
}

static std::string getAArch64TargetCPU(const llvm::Triple &triple) {
#if LDC_LLVM_VER < 1600
  auto defaultCPU = llvm::AArch64::getDefaultCPU(triple.getArchName());
  if (!defaultCPU.empty())
    return std::string(defaultCPU);
#endif

  return "generic";
}

static std::string getRiscv32TargetCPU(const llvm::Triple &triple) {
  return "generic-rv32";
}

static std::string getRiscv64TargetCPU(const llvm::Triple &triple) {
  return "generic-rv64";
}

static std::string getLoongArch32TargetCPU(const llvm::Triple &triple) {
  return "generic-la32";
}

static std::string getLoongArch64TargetCPU(const llvm::Triple &triple) {
  return "generic-la64";
}

/// Returns the LLVM name of the default CPU for the provided target triple.
static std::string getTargetCPU(const llvm::Triple &triple) {
  switch (triple.getArch()) {
  default:
    // We don't know about the specifics of this platform, just return the
    // empty string and let LLVM decide.
    return "";
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return getX86TargetCPU(triple);
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
    return getARMTargetCPU(triple);
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    return getAArch64TargetCPU(triple);
  case llvm::Triple::riscv32:
    return getRiscv32TargetCPU(triple);
  case llvm::Triple::riscv64:
    return getRiscv64TargetCPU(triple);
#if LDC_LLVM_VER >= 1600
  case llvm::Triple::loongarch32:
    return getLoongArch32TargetCPU(triple);
  case llvm::Triple::loongarch64:
    return getLoongArch64TargetCPU(triple);
#endif // LDC_LLVM_VER >= 1600
  }
}

static const char *getLLVMArchSuffixForARM(llvm::StringRef CPU) {
  return llvm::StringSwitch<const char *>(CPU)
      .Case("strongarm", "v4")
      .Cases("arm7tdmi", "arm7tdmi-s", "arm710t", "v4t")
      .Cases("arm720t", "arm9", "arm9tdmi", "v4t")
      .Cases("arm920", "arm920t", "arm922t", "v4t")
      .Cases("arm940t", "ep9312", "v4t")
      .Cases("arm10tdmi", "arm1020t", "v5")
      .Cases("arm9e", "arm926ej-s", "arm946e-s", "v5e")
      .Cases("arm966e-s", "arm968e-s", "arm10e", "v5e")
      .Cases("arm1020e", "arm1022e", "xscale", "iwmmxt", "v5e")
      .Cases("arm1136j-s", "arm1136jf-s", "arm1176jz-s", "v6")
      .Cases("arm1176jzf-s", "mpcorenovfp", "mpcore", "v6")
      .Cases("arm1156t2-s", "arm1156t2f-s", "v6t2")
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
                                     const char *llvmArchSuffix) {
  if (triple.isOSDarwin()) {
    // Darwin defaults to "softfp" for v6 and v7.
    if (llvm::StringRef(llvmArchSuffix).startswith("v6") ||
        llvm::StringRef(llvmArchSuffix).startswith("v7")) {
      return FloatABI::SoftFP;
    }
    return FloatABI::Soft;
  }

  if (triple.isOSFreeBSD()) {
    // FreeBSD defaults to soft float
    return FloatABI::Soft;
  }

  if (triple.getVendorName().startswith("hardfloat"))
    return FloatABI::Hard;
  if (triple.getVendorName().startswith("softfloat"))
    return FloatABI::SoftFP;

  switch (triple.getEnvironment()) {
  case llvm::Triple::GNUEABIHF:
    return FloatABI::Hard;
  case llvm::Triple::GNUEABI:
    return FloatABI::SoftFP;
  case llvm::Triple::EABI:
    // EABI is always AAPCS, and if it was not marked 'hard', it's softfp
    return FloatABI::SoftFP;
  case llvm::Triple::Android: {
    if (llvm::StringRef(llvmArchSuffix).startswith("v7")) {
      return FloatABI::SoftFP;
    }
    return FloatABI::Soft;
  }
  default:
    // Assume "soft".
    // TODO: Warn the user we are guessing.
    return FloatABI::Soft;
  }
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
                                 std::string &errorMsg) {
  // Allocate target machine. First, check whether the user has explicitly
  // specified an architecture to compile for. If so we have to look it up by
  // name, because it might be a backend that has no mapping to a target triple.
  const llvm::Target *target = nullptr;
  if (!arch.empty()) {
    for (const llvm::Target &T : llvm::TargetRegistry::targets()) {
      if (arch == T.getName()) {
        target = &T;
        break;
      }
    }

    if (!target) {
      errorMsg = "invalid target architecture '" + arch +
                 "', see -version for a list of supported targets.";
      return nullptr;
    }

    // Adjust the triple to match (if known), otherwise stick with the
    // given triple.
    const auto Type = llvm::Triple::getArchTypeForLLVMName(arch);
    if (Type != llvm::Triple::UnknownArch) {
      triple.setArch(Type);
      if (Type == llvm::Triple::x86)
        triple.setArchName("i686"); // instead of i386
    }
  } else {
    std::string tempError;
    target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), tempError);
    if (!target) {
      errorMsg = "unable to get target for '" + triple.getTriple() +
                 "', see -version and -mtriple.";
    }
  }

  return target;
}

llvm::TargetMachine *
createTargetMachine(const std::string targetTriple, const std::string arch,
                    std::string cpu, const std::string featuresString,
                    const ExplicitBitness::Type bitness,
                    FloatABI::Type &floatABI,
                    llvm::Optional<llvm::Reloc::Model> relocModel,
                    llvm::Optional<llvm::CodeModel::Model> codeModel,
                    const llvm::CodeGenOpt::Level codeGenOptLevel,
                    const bool noLinkerStripDead) {
  // Determine target triple. If the user didn't explicitly specify one, use
  // the one set at LLVM configure time.
  llvm::Triple triple;
  if (targetTriple.empty()) {
    triple = llvm::Triple(
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple()));

    // Apple: translate darwin to macos, apparently like clang
    if (triple.getOS() == llvm::Triple::Darwin) {
      llvm::SmallString<16> osname;
      osname += "macos";
      // We have to specify OS version in the triple to avoid linker warnings,
      // see https://github.com/ldc-developers/ldc/issues/4501.
      // If environment variable MACOSX_DEPLOYMENT_TARGET is not set, then use
      // host OS version.
      // https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/cross_development/Configuring/configuring.html
      std::string envVersion = env::get("MACOSX_DEPLOYMENT_TARGET");
      if (!envVersion.empty()) {
        osname += envVersion;
      } else {
#if LDC_LLVM_VER >= 1400
        llvm::VersionTuple OSVersion;
        triple.getMacOSXVersion(OSVersion);
        osname += OSVersion.getAsString();
#else
        // Hardcode the version, because `getMacOSXVersion` is not available.
        osname += "10.7";
#endif
      }

      triple.setOSName(osname);
    }

    // Handle -m32/-m64.
    if (!triple.isArch64Bit() && bitness == ExplicitBitness::M64) {
      triple = triple.get64BitArchVariant();
    } else if (!triple.isArch32Bit() && bitness == ExplicitBitness::M32) {
      triple = triple.get32BitArchVariant();
      if (triple.getArch() == llvm::Triple::ArchType::x86)
        triple.setArchName("i686"); // instead of i386
    }
  } else {
    triple = llvm::Triple(llvm::Triple::normalize(targetTriple));
  }

  // Look up the LLVM backend to use. This also updates triple with the
  // user-specified arch, if any.
  std::string errMsg;
  const llvm::Target *target = lookupTarget(arch, triple, errMsg);
  if (target == nullptr) {
    error(Loc(), "%s", errMsg.c_str());
    fatal();
  }

  // With an empty CPU string, LLVM will default to the host CPU, which is
  // usually not what we want (expected behavior from other compilers is
  // to default to "generic").
  if (cpu.empty() || cpu == "generic") {
    cpu = getTargetCPU(triple);
    if (cpu.empty())
      cpu = "generic";
  }

  // Package up features to be passed to target/subtarget.
  llvm::SmallVector<llvm::StringRef, 8> features;

  // NOTE: needs a persistent (non-temporary) string
  auto splitAndAddFeatures = [&features](llvm::StringRef str) {
    str.split(features, ",", -1, /* KeepEmpty */ false);
  };

  llvm::SubtargetFeatures defaultSubtargetFeatures;
  defaultSubtargetFeatures.getDefaultSubtargetFeatures(triple);
  const std::string defaultSubtargetFeaturesString =
      defaultSubtargetFeatures.getString();
  splitAndAddFeatures(defaultSubtargetFeaturesString);

  splitAndAddFeatures(featuresString);

  // checks if the features include ±<feature>
  auto hasFeature = [&features](llvm::StringRef feature) {
    return std::any_of(
        features.begin(), features.end(),
        [feature](llvm::StringRef f) { return f.substr(1) == feature; });
  };

  // cmpxchg16b is not available on old 64bit CPUs. Enable code generation
  // if the user did not make an explicit choice.
  if (cpu == "x86-64" && !hasFeature("cx16")) {
    features.push_back("+cx16");
  }

  // For a hosted RISC-V 64-bit target default to rv64gc if nothing has
  // been selected
  if (triple.getArch() == llvm::Triple::riscv64 && features.empty()) {
    const llvm::StringRef os = triple.getOSName();
    const bool isFreeStanding = os.empty() || os == "unknown" || os == "none";
    if (!isFreeStanding) {
      features = {"+m", "+a", "+f", "+d", "+c"};
    }
  }

  // For LoongArch 64-bit target default to la64 if nothing has been selected
  // All current LoongArch targets have 64-bit floating point registers.
#if LDC_LLVM_VER >= 1600
  if (triple.getArch() == llvm::Triple::loongarch64 && features.empty()) {
    features = {"+d"};
  }
#endif

  // Handle cases where LLVM picks wrong default relocModel
#if LDC_LLVM_VER >= 1600
  if (relocModel.has_value()) {}
#else
  if (relocModel.hasValue()) {}
#endif
  else {
    if (triple.isOSDarwin()) {
      // Darwin defaults to PIC (and as of 10.7.5/LLVM 3.1-3.3, TLS use leads
      // to crashes for non-PIC code). LLVM doesn't handle this.
      relocModel = llvm::Reloc::PIC_;
    } else if (triple.isOSLinux()) {
      // Modern Linux distributions have their toolchain generate PIC code for
      // additional security
      // features (like ASLR). We default to PIC code to avoid linking issues on
      // these OSes.
      // On Android, PIC is default as well.
      relocModel = llvm::Reloc::PIC_;
    } else if (triple.isOSFreeBSD()) {
      // We default to PIC code to avoid linking issues on FreeBSD, especially
      // on aarch64.
      relocModel = llvm::Reloc::PIC_;
    } else {
      // ARM for other than Darwin or Android defaults to static
      switch (triple.getArch()) {
      default:
        break;
      case llvm::Triple::arm:
      case llvm::Triple::armeb:
      case llvm::Triple::thumb:
      case llvm::Triple::thumbeb:
        relocModel = llvm::Reloc::Static;
        break;
      }
    }
  }

  llvm::TargetOptions targetOptions =
      opts::InitTargetOptionsFromCodeGenFlags(triple);

  if (targetOptions.MCOptions.ABIName.empty())
    targetOptions.MCOptions.ABIName = getABI(triple, features);

  if (floatABI == FloatABI::Default) {
    switch (triple.getArch()) {
    default: // X86, ...
      floatABI = FloatABI::Hard;
      break;
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      floatABI = getARMFloatABI(triple, getLLVMArchSuffixForARM(cpu));
      break;
    }
  }

  switch (floatABI) {
  default:
    llvm_unreachable("Floating point ABI type unknown.");
  case FloatABI::Soft:
    features.push_back("+soft-float");
    targetOptions.FloatABIType = llvm::FloatABI::Soft;
    break;
  case FloatABI::SoftFP:
    targetOptions.FloatABIType = llvm::FloatABI::Soft;
    break;
  case FloatABI::Hard:
    targetOptions.FloatABIType = llvm::FloatABI::Hard;
    break;
  }

  // Linker-level dead code elimination for ELF/wasm binaries using GNU or LLD
  // linkers (based on the --gc-sections flag) requires a separate section per
  // symbol.
  // The Apple ld64 on macOS supports a similar flag (-dead_strip) that doesn't
  // require emitting the symbols into different sections.
  // On Windows, the MSVC link.exe / lld-link.exe has `/REF`; LLD enforces
  // separate sections with LTO, so do the same here.
  if (!noLinkerStripDead && !triple.isOSBinFormatMachO()) {
    targetOptions.FunctionSections = true;
    targetOptions.DataSections = true;
  }

  // On Android, we depend on a custom TLS emulation scheme implemented in our
  // LLVM fork. LLVM 7+ enables regular emutls by default; prevent that.
  if (triple.getEnvironment() == llvm::Triple::Android) {
    targetOptions.EmulatedTLS = false;
#if LDC_LLVM_VER < 1700
    // Removed in this commit: https://github.com/llvm/llvm-project/commit/0d333bf0e3aa37e2e6ae211e3aa80631c3e01b85
    targetOptions.ExplicitEmulatedTLS = true;
#endif
  }

  const std::string finalFeaturesString =
      llvm::join(features.begin(), features.end(), ",");

  if (Logger::enabled()) {
    Logger::println("Targeting '%s' (CPU '%s' with features '%s')",
                    triple.str().c_str(), cpu.c_str(),
                    finalFeaturesString.c_str());
  }

  return target->createTargetMachine(triple.str(), cpu, finalFeaturesString,
                                     targetOptions, relocModel, codeModel,
                                     codeGenOptLevel);
}

ComputeBackend::Type getComputeTargetType(llvm::Module* m) {
  llvm::Triple::ArchType a = llvm::Triple(m->getTargetTriple()).getArch();
  if (a == llvm::Triple::spir || a == llvm::Triple::spir64)
    return ComputeBackend::SPIRV;
  else if (a == llvm::Triple::nvptx || a == llvm::Triple::nvptx64)
    return ComputeBackend::NVPTX;
  else
    return ComputeBackend::None;
}
