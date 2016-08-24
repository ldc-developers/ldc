//===-- target.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
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

/*
These guys are allocated by ddmd/target.d:
int Target::ptrsize;
int Target::realsize;
int Target::realpad;
int Target::realalignsize;
int Target::c_longsize;
int Target::c_long_doublesize;
bool Target::reverseCppOverloads;
*/

void Target::_init() {
  ptrsize = gDataLayout->getPointerSize(ADDRESS_SPACE);

  llvm::Type *real = DtoType(Type::basic[Tfloat80]);
  realsize = gDataLayout->getTypeAllocSize(real);
  realpad = realsize - gDataLayout->getTypeStoreSize(real);
  realalignsize = gDataLayout->getABITypeAlignment(real);
  c_longsize = global.params.is64bit ? 8 : 4;
  c_long_doublesize = realsize;

  // according to DMD, only for MSVC++:
  reverseCppOverloads = global.params.targetTriple->isWindowsMSVCEnvironment();

  // LDC_FIXME: Set once we support it.
  cppExceptions = false;

  RealProperties::nan = real_t::nan();
  RealProperties::snan = real_t::snan();
  RealProperties::infinity = real_t::infinity();

  auto pTargetRealSemantics = &real->getFltSemantics();
  if (pTargetRealSemantics == &APFloat::x87DoubleExtended) {
    RealProperties::max = CTFloat::parse("0x1.fffffffffffffffep+16383");
    RealProperties::min_normal = CTFloat::parse("0x1p-16382");
    RealProperties::epsilon = CTFloat::parse("0x1p-63");
    RealProperties::dig = 18;
    RealProperties::mant_dig = 64;
    RealProperties::max_exp = 16384;
    RealProperties::min_exp = -16381;
    RealProperties::max_10_exp = 4932;
    RealProperties::min_10_exp = -4932;
  } else if (pTargetRealSemantics == &APFloat::IEEEdouble ||
             pTargetRealSemantics == &APFloat::PPCDoubleDouble) {
    RealProperties::max = CTFloat::parse("0x1.fffffffffffffp+1023");
    RealProperties::min_normal = CTFloat::parse("0x1p-1022");
    RealProperties::epsilon = CTFloat::parse("0x1p-52");
    RealProperties::dig = 15;
    RealProperties::mant_dig = 53;
    RealProperties::max_exp = 1024;
    RealProperties::min_exp = -1021;
    RealProperties::max_10_exp = 308;
    RealProperties::min_10_exp = -307;
  } else {
    assert(0 && "No type properties for target `real` type");
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

// sizes based on those from tollvm.cpp:DtoMutexType()
unsigned Target::critsecsize() {
#if defined(_MSC_VER)
  // Return sizeof(RTL_CRITICAL_SECTION)
  return global.params.is64bit ? 40 : 24;
#else
  if (global.params.targetTriple->isOSWindows()) {
    return global.params.is64bit ? 40 : 24;
  }
  if (global.params.targetTriple->isOSFreeBSD() ||
#if LDC_LLVM_VER > 305
    global.params.targetTriple->isOSNetBSD() ||
    global.params.targetTriple->isOSOpenBSD() ||
    global.params.targetTriple->isOSDragonFly()
#else
    global.params.targetTriple->getOS() == llvm::Triple::NetBSD ||
    global.params.targetTriple->getOS() == llvm::Triple::OpenBSD ||
    global.params.targetTriple->getOS() == llvm::Triple::DragonFly
#endif
     ) {
    return sizeof(size_t);
  }
  return sizeof(pthread_mutex_t);

#endif
}

Type *Target::va_listType() { return gABI->vaListType(); }

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
 * Check if the given type is supported for this target
 * 0: supported
 * 1: not supported
 * 2: wrong size
 * 3: wrong base type
 */
int Target::checkVectorType(int sz, Type *type) {
  // FIXME: It is possible to query the LLVM target about supported vectors?
  return 0;
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

