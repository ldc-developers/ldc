//===-- target.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
// Implements some parts of the front-end Target class (ddmd/target.{d,h}).
//
//===----------------------------------------------------------------------===//

#include "ldcbindings.h"
#include "target.h"
#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "mars.h"
#include "mtype.h"
#include <assert.h>

#if !defined(_MSC_VER)
#include <pthread.h>
#endif

using llvm::APFloat;

// target-real values
real_t real_max;
real_t real_min_normal;
real_t real_epsilon;

// Target::RealProperties functions
real_t Target::RealProperties::nan() { return host_nan(); }
real_t Target::RealProperties::snan() { return host_snan(); }
real_t Target::RealProperties::infinity() { return host_infinity(); }
real_t Target::RealProperties::max() { return real_max; }
real_t Target::RealProperties::min_normal() { return real_min_normal; }
real_t Target::RealProperties::epsilon() { return real_epsilon; }

void Target::_init() {
  ptrsize = gDataLayout->getPointerSize(ADDRESS_SPACE);

  llvm::Type *const real = DtoType(Type::basic[Tfloat80]);
  realsize = gDataLayout->getTypeAllocSize(real);
  realpad = realsize - gDataLayout->getTypeStoreSize(real);
  realalignsize = gDataLayout->getABITypeAlignment(real);
  realislongdouble = true;

  // according to DMD, only for MSVC++:
  reverseCppOverloads = global.params.targetTriple->isWindowsMSVCEnvironment();

  // LDC_FIXME: Set once we support it.
  cppExceptions = false;

  c_longsize = global.params.is64bit ? 8 : 4;
  c_long_doublesize = realsize;
  classinfosize = 0; // unused

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

  if (targetRealSemantics == IEEEdouble) {
    real_max = CTFloat::parse("0x1.fffffffffffffp+1023");
    real_min_normal = CTFloat::parse("0x1p-1022");
    real_epsilon = CTFloat::parse("0x1p-52");
    RealProperties::dig = 15;
    RealProperties::mant_dig = 53;
    RealProperties::max_exp = 1024;
    RealProperties::min_exp = -1021;
    RealProperties::max_10_exp = 308;
    RealProperties::min_10_exp = -307;
  } else if (targetRealSemantics == x87DoubleExtended) {
    real_max = CTFloat::parse("0x1.fffffffffffffffep+16383");
    real_min_normal = CTFloat::parse("0x1p-16382");
    real_epsilon = CTFloat::parse("0x1p-63");
    RealProperties::dig = 18;
    RealProperties::mant_dig = 64;
    RealProperties::max_exp = 16384;
    RealProperties::min_exp = -16381;
    RealProperties::max_10_exp = 4932;
    RealProperties::min_10_exp = -4931;
  } else if (targetRealSemantics == IEEEquad) {
    // FIXME: hex constants
    real_max = CTFloat::parse("1.18973149535723176508575932662800702e+4932");
    real_min_normal =
        CTFloat::parse("3.36210314311209350626267781732175260e-4932");
    real_epsilon = CTFloat::parse("1.92592994438723585305597794258492732e-34");
    RealProperties::dig = 33;
    RealProperties::mant_dig = 113;
    RealProperties::max_exp = 16384;
    RealProperties::min_exp = -16381;
    RealProperties::max_10_exp = 4932;
    RealProperties::min_10_exp = -4931;
  } else {
    // rely on host compiler
    real_max = RealProperties::host_max();
    real_min_normal = RealProperties::host_min_normal();
    real_epsilon = RealProperties::host_epsilon();
    // the rest is already initialized with the corresponding real_t values
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

/******************************
 * Return size of alias Mutex in druntime/src/rt/monitor_.d, or, more precisely,
 * the size of the native critical section as 2nd field in struct
 * D_CRITICAL_SECTION (after a pointer). D_CRITICAL_SECTION is pointer-size
 * aligned, so the returned field size is a multiple of pointer-size.
 */
unsigned Target::critsecsize() {
  const bool is64bit = global.params.is64bit;

  // Windows: sizeof(CRITICAL_SECTION)
  if (global.params.isWindows)
    return is64bit ? 40 : 24;

  // POSIX: sizeof(pthread_mutex_t)
  // based on druntime/src/core/sys/posix/sys/types.d
  const auto &triple = *global.params.targetTriple;
  const auto arch = triple.getArch();
  switch (triple.getOS()) {
  case llvm::Triple::Linux:
    if (triple.getEnvironment() == llvm::Triple::Android)
      return Target::ptrsize; // 32-bit integer rounded up to pointer size
    if (arch == llvm::Triple::aarch64 || arch == llvm::Triple::aarch64_be)
      return 48;
    return is64bit ? 40 : 24;

  case llvm::Triple::MacOSX:
    return is64bit ? 64 : 44;

  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
  case llvm::Triple::DragonFly:
    return Target::ptrsize;

  case llvm::Triple::Solaris:
    return 24;

  default:
    break;
  }

#ifndef _MSC_VER
  unsigned hostSize = sizeof(pthread_mutex_t);
  warning(Loc(), "Assuming critical section size = %u bytes", hostSize);
  return hostSize;
#else
  error(Loc(), "Unknown critical section size");
  fatal();
  return 0;
#endif
}

Type *Target::va_listType() { return gABI->vaListType(); }

/******************************
 * Check if the given type is supported for this target
 * 0: supported
 * 1: not supported
 * 2: wrong size
 * 3: wrong base type
 */
int Target::checkVectorType(int sz, Type *type) {
  // FIXME: Is it possible to query the LLVM target about supported vectors?
  return 0;
}

/******************************
 * Encode the given expression, which is assumed to be an rvalue literal
 * as another type for use in CTFE.
 * This corresponds roughly to the idiom *(Type *)&e.
 */
Expression *Target::paintAsType(Expression *e, Type *type) {
  union {
    d_int32 int32value;
    d_int64 int64value;
    float float32value;
    double float64value;
  } u;

  assert(e->type->size() == type->size());

  switch (e->type->ty) {
  case Tint32:
  case Tuns32:
    u.int32value = static_cast<d_int32>(e->toInteger());
    break;

  case Tint64:
  case Tuns64:
    u.int64value = static_cast<d_int64>(e->toInteger());
    break;

  case Tfloat32:
    u.float32value = e->toReal();
    break;

  case Tfloat64:
    u.float64value = e->toReal();
    break;

  default:
    assert(0);
  }

  switch (type->ty) {
  case Tint32:
  case Tuns32:
    return createIntegerExp(e->loc, u.int32value, type);

  case Tint64:
  case Tuns64:
    return createIntegerExp(e->loc, u.int64value, type);

  case Tfloat32:
    return createRealExp(e->loc, u.float32value, type);

  case Tfloat64:
    return createRealExp(e->loc, u.float64value, type);

  default:
    assert(0);
  }

  return nullptr; // avoid warning
}

/******************************
 * For the given module, perform any post parsing analysis.
 * Certain compiler backends (ie: GDC) have special placeholder
 * modules whose source are empty, but code gets injected
 * immediately after loading.
 */
void Target::loadModule(Module *m) {}

/******************************
 *
 */
void Target::prefixName(OutBuffer *buf, LINK linkage) {}
