//===-- structs.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/structs.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/init.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"
#include "ir/irdsymbol.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////

void DtoResolveStruct(StructDeclaration *sd) { DtoResolveStruct(sd, sd->loc); }

void DtoResolveStruct(StructDeclaration *sd, const Loc &callerLoc) {
  // Make sure to resolve each struct type exactly once.
  if (sd->ir->isResolved()) {
    return;
  }
  sd->ir->setResolved();

  IF_LOG Logger::println("Resolving struct type: %s (%s)", sd->toChars(),
                         sd->loc.toChars());
  LOG_SCOPE;

  // make sure type exists
  DtoType(sd->type);

  // if it's a forward declaration, all bets are off. The type should be enough
  if (sd->sizeok != Sizeok::done) {
    error(callerLoc, "struct `%s.%s` unknown size", sd->getModule()->toChars(),
          sd->toChars());
    fatal();
  }

  // create the IrAggr
  getIrAggr(sd, true);

  // Set up our field metadata.
  for (auto vd : sd->fields) {
    IF_LOG {
      if (isIrFieldCreated(vd)) {
        Logger::println("struct field already exists");
      }
    }
    getIrField(vd, true);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////   D STRUCT UTILITIES
///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

LLValue *DtoStructEquals(EXP op, DValue *lhs, DValue *rhs) {
  Type *t = lhs->type->toBasetype();
  assert(t->ty == TY::Tstruct);

  // set predicate
  llvm::ICmpInst::Predicate cmpop;
  if (op == EXP::equal || op == EXP::identity) {
    cmpop = llvm::ICmpInst::ICMP_EQ;
  } else {
    cmpop = llvm::ICmpInst::ICMP_NE;
  }

  // empty struct? EQ always true, NE always false
  if (static_cast<TypeStruct *>(t)->sym->fields.length == 0) {
    return DtoConstBool(cmpop == llvm::ICmpInst::ICMP_EQ);
  }

  // call memcmp
  size_t sz = getTypeAllocSize(DtoType(t));
  LLValue *val = DtoMemCmp(DtoLVal(lhs), DtoLVal(rhs), DtoConstSize_t(sz));
  return gIR->ir->CreateICmp(cmpop, val,
                             LLConstantInt::get(val->getType(), 0, false));
}

////////////////////////////////////////////////////////////////////////////////

/// Return the type returned by DtoUnpaddedStruct called on a value of the
/// specified type.
/// Union types will get expanded into a struct, with a type for each member.
LLType *DtoUnpaddedStructType(Type *dty) {
  assert(dty->ty == TY::Tstruct);

  typedef llvm::DenseMap<Type *, llvm::StructType *> CacheT;
  static llvm::ManagedStatic<CacheT> cache;
  auto it = cache->find(dty);
  if (it != cache->end()) {
    return it->second;
  }

  TypeStruct *sty = static_cast<TypeStruct *>(dty);
  VarDeclarations &fields = sty->sym->fields;

  std::vector<LLType *> types;
  types.reserve(fields.length);

  for (unsigned i = 0; i < fields.length; i++) {
    LLType *fty;
    if (fields[i]->type->ty == TY::Tstruct) {
      // Nested structs are the only members that can contain padding
      fty = DtoUnpaddedStructType(fields[i]->type);
    } else {
      fty = DtoType(fields[i]->type);
    }
    types.push_back(fty);
  }
  LLStructType *Ty = LLStructType::get(gIR->context(), types);
  cache->insert(std::make_pair(dty, Ty));
  return Ty;
}

/// Return the struct value represented by v without the padding fields.
/// Unions will be expanded, with a value for each member.
/// Note: v must be a pointer to a struct, but the return value will be a
///       first-class struct value.
LLValue *DtoUnpaddedStruct(Type *dty, LLValue *v) {
  assert(dty->ty == TY::Tstruct);
  TypeStruct *sty = static_cast<TypeStruct *>(dty);
  VarDeclarations &fields = sty->sym->fields;

  LLValue *newval = llvm::UndefValue::get(DtoUnpaddedStructType(dty));

  for (unsigned i = 0; i < fields.length; i++) {
    LLValue *fieldptr = DtoLVal(DtoIndexAggregate(v, sty->sym, fields[i]));
    LLValue *fieldval;
    if (fields[i]->type->ty == TY::Tstruct) {
      // Nested structs are the only members that can contain padding
      fieldval = DtoUnpaddedStruct(fields[i]->type, fieldptr);
    } else {
      assert(!fields[i]->isBitFieldDeclaration());
      fieldval = DtoLoad(DtoType(fields[i]->type), fieldptr);
    }
    newval = DtoInsertValue(newval, fieldval, i);
  }
  return newval;
}

/// Undo the transformation performed by DtoUnpaddedStruct, writing to lval.
void DtoPaddedStruct(Type *dty, LLValue *v, LLValue *lval) {
  assert(dty->ty == TY::Tstruct);
  TypeStruct *sty = static_cast<TypeStruct *>(dty);
  VarDeclarations &fields = sty->sym->fields;

  for (unsigned i = 0; i < fields.length; i++) {
    LLValue *fieldptr = DtoLVal(DtoIndexAggregate(lval, sty->sym, fields[i]));
    LLValue *fieldval = DtoExtractValue(v, i);
    if (fields[i]->type->ty == TY::Tstruct) {
      // Nested structs are the only members that can contain padding
      DtoPaddedStruct(fields[i]->type, fieldval, fieldptr);
    } else {
      assert(!fields[i]->isBitFieldDeclaration());
      DtoStoreZextI8(fieldval, fieldptr);
    }
  }
}
