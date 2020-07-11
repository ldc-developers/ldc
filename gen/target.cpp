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

#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "driver/cl_options.h"
#include "driver/linker.h"
#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include <assert.h>

using llvm::APFloat;

// in dmd/argtypes.d:
TypeTuple *toArgTypes(Type *t);
// in dmd/argtypes_sysv_x64.d:
TypeTuple *toArgTypes_sysv_x64(Type *t);
// in dmd/argtypes_aarch64.d:
TypeTuple *toArgTypes_aarch64(Type *t);

namespace {
/******************************
 * Return size of alias Mutex in druntime/src/rt/monitor_.d, or, more precisely,
 * the size of the native critical section as 2nd field in struct
 * D_CRITICAL_SECTION (after a pointer). D_CRITICAL_SECTION is pointer-size
 * aligned, so the returned field size is a multiple of pointer-size.
 */
unsigned getCriticalSectionSize(const Param &params) {
  const bool is64bit = params.is64bit;

  // Windows: sizeof(CRITICAL_SECTION)
  if (params.isWindows)
    return is64bit ? 40 : 24;

  // POSIX: sizeof(pthread_mutex_t)
  // based on druntime/src/core/sys/posix/sys/types.d
  const auto &triple = *params.targetTriple;

  if (triple.isOSDarwin())
    return is64bit ? 64 : 44;

  const auto arch = triple.getArch();
  switch (triple.getOS()) {
  case llvm::Triple::Linux:
    if (triple.getEnvironment() == llvm::Triple::Android) {
      // 32-bit integer rounded up to pointer size
      return gDataLayout->getPointerSize();
    }
    if (arch == llvm::Triple::aarch64 || arch == llvm::Triple::aarch64_be)
      return 48;
    return is64bit ? 40 : 24;

  case llvm::Triple::NetBSD:
    return is64bit ? 48 : 28;

  case llvm::Triple::FreeBSD:
  case llvm::Triple::OpenBSD:
  case llvm::Triple::DragonFly:
    return gDataLayout->getPointerSize();

  case llvm::Triple::Solaris:
    return 24;

  default:
    return 0; // leads to an error whenever requested
  }
}
} // anonymous namespace

void Target::_init(const Param &params) {
  CTFloat::initialize();
  initFPTypeProperties();

  const auto &triple = *params.targetTriple;
  const bool isMSVC = triple.isWindowsMSVCEnvironment();
  llvm::Type *const real = DtoType(Type::basic[Tfloat80]);

  ptrsize = gDataLayout->getPointerSize();
  realsize = gDataLayout->getTypeAllocSize(real);
  realpad = realsize - gDataLayout->getTypeStoreSize(real);
  realalignsize = gDataLayout->getABITypeAlignment(real);
  classinfosize = 0; // unused
  maxStaticDataSize = std::numeric_limits<unsigned long long>::max();

  c.longsize = params.is64bit && !isMSVC ? 8 : 4;
  c.long_doublesize = realsize;
  c.criticalSectionSize = getCriticalSectionSize(params);

  cpp.reverseOverloads = isMSVC; // according to DMD, only for MSVC++
  cpp.exceptions = true;
  cpp.twoDtorInVtable = !isMSVC;

  objc.supported = params.hasObjectiveC;

  // Finalize RealProperties for the target's `real` type.

  const auto targetRealSemantics = &real->getFltSemantics();
#if LDC_LLVM_VER >= 400
  const auto IEEEdouble = &APFloat::IEEEdouble();
  const auto x87DoubleExtended = &APFloat::x87DoubleExtended();
  const auto IEEEquad = &APFloat::IEEEquad();
#else
  const auto IEEEdouble = &APFloat::IEEEdouble;
  const auto x87DoubleExtended = &APFloat::x87DoubleExtended;
  const auto IEEEquad = &APFloat::IEEEquad;
#endif

  RealProperties.nan = CTFloat::nan;
  RealProperties.infinity = CTFloat::infinity;

  if (targetRealSemantics == IEEEdouble) {
    RealProperties.max = CTFloat::parseReal("0x1.fffffffffffffp+1023");
    RealProperties.min_normal = CTFloat::parseReal("0x1p-1022");
    RealProperties.epsilon = CTFloat::parseReal("0x1p-52");
    RealProperties.dig = 15;
    RealProperties.mant_dig = 53;
    RealProperties.max_exp = 1024;
    RealProperties.min_exp = -1021;
    RealProperties.max_10_exp = 308;
    RealProperties.min_10_exp = -307;
  } else if (targetRealSemantics == x87DoubleExtended) {
    RealProperties.max = CTFloat::parseReal("0x1.fffffffffffffffep+16383");
    RealProperties.min_normal = CTFloat::parseReal("0x1p-16382");
    RealProperties.epsilon = CTFloat::parseReal("0x1p-63");
    RealProperties.dig = 18;
    RealProperties.mant_dig = 64;
    RealProperties.max_exp = 16384;
    RealProperties.min_exp = -16381;
    RealProperties.max_10_exp = 4932;
    RealProperties.min_10_exp = -4931;
  } else if (targetRealSemantics == IEEEquad) {
    // FIXME: hex constants
    RealProperties.max =
        CTFloat::parseReal("1.18973149535723176508575932662800702e+4932");
    RealProperties.min_normal =
        CTFloat::parseReal("3.36210314311209350626267781732175260e-4932");
    RealProperties.epsilon =
        CTFloat::parseReal("1.92592994438723585305597794258492732e-34");
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
  if (type->ty == Tvoid) {
    return 1;
  }
  return gDataLayout->getABITypeAlignment(DtoType(type));
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
  if (t->ty == Tfloat80) {
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
    return ::toArgTypes(t);
  if (arch == llvm::Triple::x86_64 && !triple.isOSWindows())
    return toArgTypes_sysv_x64(t);
  if (arch == llvm::Triple::aarch64 || arch == llvm::Triple::aarch64_be)
    return toArgTypes_aarch64(t);
  return nullptr;
}

bool Target::isReturnOnStack(TypeFunction *tf, bool needsThis) {
  return gABI->returnInArg(tf, needsThis);
}

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
#if LDC_LLVM_VER >= 500
    } else if (triple.isOSBinFormatWasm()) {
      objectFormat = "wasm";
#endif
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

  if (name == "cppStd")
    return createIntegerExp(static_cast<unsigned>(global.params.cplusplus));

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
  if (name == "dcomputeTargets") {
    Expressions* exps = new Expressions();
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
