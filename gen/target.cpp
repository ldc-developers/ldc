//===-- target.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
// Implements some parts of the front-end Target class (dmd/target.{d,h}).
//
//===----------------------------------------------------------------------===//

#include "dmd/argtypes.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/ldcbindings.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/linker.h"
#include "gen/abi/abi.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include <assert.h>

using namespace dmd;
using llvm::APFloat;

namespace {
// Returns the LL type to be used for D `real` (C `long double`).
llvm::Type *getRealType(const llvm::Triple &triple) {
  using llvm::Triple;

  auto &ctx = getGlobalContext();

  // Android: x86 targets follow ARM, with emulated quad precision for x64
  if (triple.getEnvironment() == llvm::Triple::Android) {
    return triple.isArch64Bit() ? LLType::getFP128Ty(ctx)
                                : LLType::getDoubleTy(ctx);
  }

  switch (triple.getArch()) {
  case Triple::x86:
  case Triple::x86_64:
    // only x86 has 80-bit extended precision; MSVC uses double
    return triple.isWindowsMSVCEnvironment() ? LLType::getDoubleTy(ctx)
                                             : LLType::getX86_FP80Ty(ctx);

  case Triple::aarch64:
  case Triple::aarch64_be:
    // AArch64 has 128-bit quad precision; Apple uses double
    return triple.isOSDarwin() ? LLType::getDoubleTy(ctx)
                               : LLType::getFP128Ty(ctx);

  case Triple::riscv32:
  case Triple::riscv64:
#if LDC_LLVM_VER >= 1600
  case Triple::loongarch32:
  case Triple::loongarch64:
#endif // LDC_LLVM_VER >= 1600
    return LLType::getFP128Ty(ctx);

  default:
    // 64-bit double precision for all other targets
    // FIXME: PowerPC, SystemZ, ...
    return LLType::getDoubleTy(ctx);
  }
}
}

void Target::_init(const Param &params) {
  this->params = &params;

  CTFloat::initialize();
  initFPTypeProperties();

  const auto &triple = *params.targetTriple;
  const bool isMSVC = triple.isWindowsMSVCEnvironment();

  if (triple.isOSLinux()) {
    os = OS_linux;
  } else if (triple.isOSDarwin()) {
    os = OS_OSX;
  } else if (triple.isOSWindows()) {
    os = OS_Windows;
  } else if (triple.isOSFreeBSD()) {
    os = OS_FreeBSD;
  } else if (triple.isOSOpenBSD()) {
    os = OS_OpenBSD;
  } else if (triple.isOSDragonFly()) {
    os = OS_DragonFlyBSD;
  } else if (triple.isOSSolaris()) {
    os = OS_Solaris;
  } else {
    os = OS_Freestanding;
  }

  osMajor = triple.getOSMajorVersion();

  ptrsize = gDataLayout->getPointerSize();
  realType = getRealType(triple);
  realsize = gDataLayout->getTypeAllocSize(realType);
  realpad = realsize - gDataLayout->getTypeStoreSize(realType);
  realalignsize = gDataLayout->getABITypeAlign(realType).value();
  classinfosize = 0; // unused
  maxStaticDataSize = std::numeric_limits<unsigned long long>::max();

  c.crtDestructorsSupported = true; // unused as of 2.099
  c.boolsize = 1;
  c.shortsize = 2;
  c.intsize = 4;
  c.longsize = (ptrsize == 8) && !isMSVC ? 8 : 4;
  c.long_longsize = 8;
  c.long_doublesize = realsize;
  c.wchar_tsize = triple.isOSWindows() ? 2 : 4;
  c.bitFieldStyle =
      isMSVC ? TargetC::BitFieldStyle::MS : TargetC::BitFieldStyle::Gcc_Clang;

  cpp.reverseOverloads = isMSVC; // according to DMD, only for MSVC++
  cpp.exceptions = true;
  cpp.twoDtorInVtable = !isMSVC;
  cpp.splitVBasetable = isMSVC;
  cpp.wrapDtorInExternD = triple.getArch() == llvm::Triple::x86;

  objc.supported = objc_isSupported(triple);

  const llvm::StringRef archName = triple.getArchName();
  architectureName = {archName.size(), archName.data()};

  isLP64 = gDataLayout->getPointerSizeInBits() == 64;
  run_noext = !triple.isOSWindows();

  if (isMSVC) {
    obj_ext = {3, "obj"};
    lib_ext = {3, "lib"};
  } else {
    obj_ext = {1, "o"};
    lib_ext = {1, "a"};
  }

  if (triple.isOSWindows()) {
    dll_ext = {3, "dll"};
  } else if (triple.isOSDarwin()) {
    dll_ext = {5, "dylib"};
  } else {
    dll_ext = {2, "so"};
  }

  // Finalize RealProperties for the target's `real` type.

  const auto targetRealSemantics = &realType->getFltSemantics();
  const auto IEEEdouble = &APFloat::IEEEdouble();
  const auto x87DoubleExtended = &APFloat::x87DoubleExtended();
  const auto IEEEquad = &APFloat::IEEEquad();
  bool isOutOfRange = false;

  RealProperties.nan = CTFloat::nan;
  RealProperties.infinity = CTFloat::infinity;

  if (targetRealSemantics == IEEEdouble) {
    RealProperties.max =
        CTFloat::parse("0x1.fffffffffffffp+1023", isOutOfRange);
    RealProperties.min_normal = CTFloat::parse("0x1p-1022", isOutOfRange);
    RealProperties.epsilon = CTFloat::parse("0x1p-52", isOutOfRange);
    RealProperties.dig = 15;
    RealProperties.mant_dig = 53;
    RealProperties.max_exp = 1024;
    RealProperties.min_exp = -1021;
    RealProperties.max_10_exp = 308;
    RealProperties.min_10_exp = -307;
  } else if (targetRealSemantics == x87DoubleExtended) {
    RealProperties.max =
        CTFloat::parse("0x1.fffffffffffffffep+16383", isOutOfRange);
    RealProperties.min_normal = CTFloat::parse("0x1p-16382", isOutOfRange);
    RealProperties.epsilon = CTFloat::parse("0x1p-63", isOutOfRange);
    RealProperties.dig = 18;
    RealProperties.mant_dig = 64;
    RealProperties.max_exp = 16384;
    RealProperties.min_exp = -16381;
    RealProperties.max_10_exp = 4932;
    RealProperties.min_10_exp = -4931;
  } else if (targetRealSemantics == IEEEquad) {
    // FIXME: hex constants
    RealProperties.max = CTFloat::parse(
        "1.18973149535723176508575932662800702e+4932", isOutOfRange);
    RealProperties.min_normal = CTFloat::parse(
        "3.36210314311209350626267781732175260e-4932", isOutOfRange);
    RealProperties.epsilon = CTFloat::parse(
        "1.92592994438723585305597794258492732e-34", isOutOfRange);
    RealProperties.dig = 33;
    RealProperties.mant_dig = 113;
    RealProperties.max_exp = 16384;
    RealProperties.min_exp = -16381;
    RealProperties.max_10_exp = 4932;
    RealProperties.min_10_exp = -4931;
  } else {
    // leave initialized with host real_t values
    warning(Loc(), "unknown properties for target `real` type, relying on D "
                   "host compiler");
  }
}

/******************************
 * Return memory alignment size of type.
 */
unsigned Target::alignsize(Type *type) {
  assert(type->isTypeBasic());
  if (type->ty == TY::Tvoid) {
    return 1;
  }
  return gDataLayout->getABITypeAlign(DtoType(type)).value();
}

/******************************
 * Return field alignment size of type.
 */
unsigned Target::fieldalign(Type *type) { return DtoAlignment(type); }

Type *Target::va_listType(const Loc &loc, Scope *sc) {
  if (!tvalist)
    tvalist = typeSemantic(gABI->vaListType(), loc, sc);
  return tvalist;
}

/**
 * Gets vendor-specific type mangling for C++ ABI.
 * Params:
 *      t = type to inspect
 * Returns:
 *      string if type is mangled specially on target
 *      null if unhandled
 */
const char *TargetCPP::typeMangle(Type *t) {
  if (t->ty == TY::Tfloat80) {
    const auto &triple = *global.params.targetTriple;
    // `long double` on Android/x64 is __float128 and mangled as `g`
    bool isAndroidX64 = triple.getEnvironment() == llvm::Triple::Android &&
                        triple.getArch() == llvm::Triple::x86_64;
    return isAndroidX64 ? "g" : "e";
  }
  return nullptr;
}

TypeTuple *Target::toArgTypes(Type *t) {
  const auto &triple = *global.params.targetTriple;
  const auto arch = triple.getArch();
  if (arch == llvm::Triple::x86)
    return toArgTypes_x86(t);
  if (arch == llvm::Triple::x86_64 && !triple.isOSWindows())
    return toArgTypes_sysv_x64(t);
  if (arch == llvm::Triple::aarch64 || arch == llvm::Triple::aarch64_be)
    return toArgTypes_aarch64(t);
  return nullptr;
}

bool Target::isReturnOnStack(TypeFunction *tf, bool needsThis) {
  return gABI->returnInArg(tf, needsThis);
}

bool Target::preferPassByRef(Type *t) { return gABI->preferPassByRef(t); }

Expression *Target::getTargetInfo(const char *name_, const Loc &loc) {
  const llvm::StringRef name(name_);
  const auto &triple = *global.params.targetTriple;

  const auto createStringExp = [&loc](const char *value) {
    return value ? StringExp::create(loc, const_cast<char *>(value)) : nullptr;
  };

  if (name == "objectFormat") {
    const char *objectFormat = nullptr;
    if (triple.isOSBinFormatCOFF()) {
      objectFormat = "coff";
    } else if (triple.isOSBinFormatMachO()) {
      objectFormat = "macho";
    } else if (triple.isOSBinFormatELF()) {
      objectFormat = "elf";
    } else if (triple.isOSBinFormatWasm()) {
      objectFormat = "wasm";
    }
    return createStringExp(objectFormat);
  }

  if (name == "floatAbi") {
    const char *floatAbi = nullptr;
    if (opts::floatABI == FloatABI::Hard) {
      floatAbi = "hard";
    } else if (opts::floatABI == FloatABI::Soft) {
      floatAbi = "soft";
    } else if (opts::floatABI == FloatABI::SoftFP) {
      floatAbi = "softfp";
    }
    return createStringExp(floatAbi);
  }

  if (name == "cppRuntimeLibrary") {
    const char *cppRuntimeLibrary = "";
    if (triple.isWindowsMSVCEnvironment()) {
      auto mscrtlib = getMscrtLibName().str();
      cppRuntimeLibrary = mem.xstrdup(mscrtlib.c_str());
    }
    return createStringExp(cppRuntimeLibrary);
  }

  if (name == "cppStd") {
    return IntegerExp::create(
        Loc(), static_cast<unsigned>(global.params.cplusplus), Type::tint32);
  }

  if (name == "CET") {
    auto cet = opts::fCFProtection.getValue();
    return IntegerExp::create(loc, static_cast<unsigned>(cet), Type::tint32);
  }

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
  if (name == "dcomputeTargets") {
    Expressions* exps = createExpressions();
    for (auto &targ : opts::dcomputeTargets) {
        exps->push(createStringExp(mem.xstrdup(targ.c_str())));
    }
    return TupleExp::create(loc, exps);
  }

  if (name == "dcomputeFilePrefix") {
    return createStringExp(
                mem.xstrdup(opts::dcomputeFilePrefix.c_str()));
  }
#endif

  return nullptr;
}

bool Target::isCalleeDestroyingArgs(TypeFunction* tf) {
  // callEE for extern(D) and MSVC++; callER for non-MSVC extern(C++)
  return global.params.targetTriple->isWindowsMSVCEnvironment() ||
         tf->linkage != LINK::cpp;
}

bool Target::supportsLinkerDirective() const {
  return global.params.targetTriple->isWindowsMSVCEnvironment() ||
         global.params.targetTriple->isOSBinFormatMachO();
}
