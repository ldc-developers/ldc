//===-- toconstelem.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/ctfe.h"
#include "dmd/errors.h"
#include "dmd/template.h"
#include "gen/arrays.h"
#include "gen/binops.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"

/// Emits an LLVM constant corresponding to the expression.
///
/// Due to the current implementation of AssocArrayLiteralExp::toElem, the
/// implementations have to be able to handle being called on expressions
/// that are not actually constant. In such a case, an LLVM undef of the
/// expected type should be returned (_not_ null).
class ToConstElemVisitor : public Visitor {
  IRState *p;
  LLConstant *result;

public:
  explicit ToConstElemVisitor(IRState *p_) : p(p_) {}

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////////

  LLConstant *toConstElem(Expression *e) {
    result = nullptr;
    e->accept(this);
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////

  void fatalError(Expression *e) {
    if (!global.gag) {
      fatal();
    }

    // Do not return null here, as AssocArrayLiteralExp::toElem determines
    // whether it can allocate the needed arrays statically by just invoking
    // toConstElem on its key/value expressions, and handling the null value
    // consequently would require error-prone adaptions in all other code.
    result = llvm::UndefValue::get(DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(VarExp *e) override {
    IF_LOG Logger::print("VarExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    if (SymbolDeclaration *sdecl = e->var->isSymbolDeclaration()) {
      // this seems to be the static initialiser for structs
      Type *sdecltype = sdecl->type->toBasetype();
      IF_LOG Logger::print("Sym: type=%s\n", sdecltype->toChars());
      assert(sdecltype->ty == Tstruct);
      TypeStruct *ts = static_cast<TypeStruct *>(sdecltype);
      DtoResolveStruct(ts->sym);
      result = getIrAggr(ts->sym)->getDefaultInit();
      return;
    }

    if (TypeInfoDeclaration *ti = e->var->isTypeInfoDeclaration()) {
      result = DtoTypeInfoOf(ti->tinfo, false);
      result = DtoBitCast(result, DtoType(e->type));
      return;
    }

    VarDeclaration *vd = e->var->isVarDeclaration();
    if (vd && vd->isConst() && vd->_init) {
      if (vd->inuse) {
        e->error("recursive reference `%s`", e->toChars());
        result = llvm::UndefValue::get(DtoType(e->type));
      } else {
        vd->inuse++;
        // return the initializer
        result = DtoConstInitializer(e->loc, e->type, vd->_init);
        vd->inuse--;
      }
    }
    // fail
    else {
      e->error("non-constant expression `%s`", e->toChars());
      result = llvm::UndefValue::get(DtoType(e->type));
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(IntegerExp *e) override {
    IF_LOG Logger::print("IntegerExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    LLType *t = DtoType(e->type);
    if (isaPointer(t)) {
      Logger::println("pointer");
      LLConstant *i = LLConstantInt::get(
          DtoSize_t(), static_cast<uint64_t>(e->getInteger()), false);
      result = llvm::ConstantExpr::getIntToPtr(i, t);
    } else {
      assert(llvm::isa<LLIntegerType>(t));
      result = LLConstantInt::get(t, static_cast<uint64_t>(e->getInteger()),
                                  !e->type->isunsigned());
      assert(result);
      IF_LOG Logger::cout() << "value = " << *result << '\n';
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(RealExp *e) override {
    IF_LOG Logger::print("RealExp::toConstElem: %s @ %s | %La\n", e->toChars(),
                         e->type->toChars(), e->value);
    LOG_SCOPE;
    Type *t = e->type->toBasetype();
    result = DtoConstFP(t, e->value);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(NullExp *e) override {
    IF_LOG Logger::print("NullExp::toConstElem(type=%s): %s\n",
                         e->type->toChars(), e->toChars());
    LOG_SCOPE;
    LLType *t = DtoType(e->type);
    if (e->type->ty == Tarray) {
      assert(isaStruct(t));
      result = llvm::ConstantAggregateZero::get(t);
    } else {
      result = LLConstant::getNullValue(t);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ComplexExp *e) override {
    IF_LOG Logger::print("ComplexExp::toConstElem(): %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    result = DtoConstComplex(e->type, e->value.re, e->value.im);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(StringExp *e) override {
    IF_LOG Logger::print("StringExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    Type *const t = e->type->toBasetype();

    if (t->ty == Tsarray) {
      result = buildStringLiteralConstant(e, false);
      return;
    }

    llvm::GlobalVariable *gvar = p->getCachedStringLiteral(e);

    llvm::ConstantInt *zero =
        LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant *idxs[2] = {zero, zero};
    LLConstant *arrptr = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(gvar)->getElementType(), gvar, idxs, true);

    if (t->ty == Tpointer) {
      result = arrptr;
    } else if (t->ty == Tarray) {
      LLConstant *clen =
          LLConstantInt::get(DtoSize_t(), e->numberOfCodeUnits(), false);
      result = DtoConstSlice(clen, arrptr, e->type);
    } else {
      llvm_unreachable("Unknown type for StringExp.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AddExp *e) override {
    IF_LOG Logger::print("AddExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // add to pointer
    Type *t1b = e->e1->type->toBasetype();
    if (t1b->ty == Tpointer && e->e2->type->isintegral()) {
      llvm::Constant *ptr = toConstElem(e->e1);
      dinteger_t idx = undoStrideMul(e->loc, t1b, e->e2->toInteger());
      result = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(ptr)->getElementType(), ptr, DtoConstSize_t(idx));
      return;
    }

    visit(static_cast<Expression *>(e));
  }

  void visit(MinExp *e) override {
    IF_LOG Logger::print("MinExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    Type *t1b = e->e1->type->toBasetype();
    if (t1b->ty == Tpointer && e->e2->type->isintegral()) {
      llvm::Constant *ptr = toConstElem(e->e1);
      dinteger_t idx = undoStrideMul(e->loc, t1b, e->e2->toInteger());

      llvm::Constant *negIdx = llvm::ConstantExpr::getNeg(DtoConstSize_t(idx));
      result = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(ptr)->getElementType(), ptr, negIdx);
      return;
    }

    visit(static_cast<Expression *>(e));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(CastExp *e) override {
    IF_LOG Logger::print("CastExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    LLType *lltype = DtoType(e->type);
    Type *tb = e->to->toBasetype();

    // string literal to dyn array:
    // reinterpret the string data as an array, calculate the length
    if (e->e1->op == TOKstring && tb->ty == Tarray) {
#if 0
            StringExp *strexp = static_cast<StringExp*>(e1);
            size_t datalen = strexp->sz * strexp->len;
            Type* eltype = tb->nextOf()->toBasetype();
            if (datalen % eltype->size() != 0) {
                error("the sizes don't line up");
                return e1->toConstElem(p);
            }
            size_t arrlen = datalen / eltype->size();
#endif
      e->error("ct cast of `string` to dynamic array not fully implemented");
      result = toConstElem(e->e1);
    }
    // pointer to pointer
    else if (tb->ty == Tpointer && e->e1->type->toBasetype()->ty == Tpointer) {
      result = llvm::ConstantExpr::getBitCast(toConstElem(e->e1), lltype);
    }
    // global variable to pointer
    else if (tb->ty == Tpointer && e->e1->op == TOKvar) {
      VarDeclaration *vd =
          static_cast<VarExp *>(e->e1)->var->isVarDeclaration();
      assert(vd);
      DtoResolveVariable(vd);
      LLConstant *value =
          isIrGlobalCreated(vd) ? isaConstant(getIrGlobal(vd)->value) : nullptr;
      if (!value) {
        goto Lerr;
      }
      Type *type = vd->type->toBasetype();
      if (type->ty == Tarray || type->ty == Tdelegate) {
        LLConstant *idxs[2] = {DtoConstSize_t(0), DtoConstSize_t(1)};
        value = llvm::ConstantExpr::getGetElementPtr(
            isaPointer(value)->getElementType(), value, idxs, true);
      }
      result = DtoBitCast(value, DtoType(tb));
    } else if (tb->ty == Tclass && e->e1->type->ty == Tclass &&
               e->e1->op == TOKclassreference) {
      auto cd = static_cast<ClassReferenceExp *>(e->e1)->originalClass();
      llvm::Constant *instance = toConstElem(e->e1);
      if (InterfaceDeclaration *it =
              static_cast<TypeClass *>(tb)->sym->isInterfaceDeclaration()) {
        assert(it->isBaseOf(cd, NULL));

        IrTypeClass *typeclass = cd->type->ctype->isClass();

        // find interface impl
        size_t i_index = typeclass->getInterfaceIndex(it);
        assert(i_index != ~0UL);

        // offset pointer
        instance = DtoGEP(instance, 0, i_index);
      }
      result = DtoBitCast(instance, DtoType(tb));
    } else {
      goto Lerr;
    }
    return;

  Lerr:
    e->error("cannot cast `%s` to `%s` at compile time", e->e1->type->toChars(),
             e->type->toChars());
    fatalError(e);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(SymOffExp *e) override {
    IF_LOG Logger::println("SymOffExp::toConstElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;

    llvm::Constant *base = DtoConstSymbolAddress(e->loc, e->var);
    if (base == nullptr) {
      result = llvm::UndefValue::get(DtoType(e->type));
      return;
    }

    if (e->offset == 0) {
      result = base;
    } else {
      const unsigned elemSize =
          gDataLayout->getTypeStoreSize(base->getType()->getContainedType(0));

      IF_LOG Logger::println("adding offset: %llu (elem size: %u)",
                             static_cast<unsigned long long>(e->offset),
                             elemSize);

      if (e->offset % elemSize == 0) {
        // We can turn this into a "nice" GEP.
        result = llvm::ConstantExpr::getGetElementPtr(nullptr,
            base, DtoConstSize_t(e->offset / elemSize));
      } else {
        // Offset isn't a multiple of base type size, just cast to i8* and
        // apply the byte offset.
        result = llvm::ConstantExpr::getGetElementPtr(nullptr,
            DtoBitCast(base, getVoidPtrType()), DtoConstSize_t(e->offset));
      }
    }

    result = DtoBitCast(result, DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(AddrExp *e) override {
    IF_LOG Logger::println("AddrExp::toConstElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;
    // FIXME: this should probably be generalized more so we don't
    // need to have a case for each thing we can take the address of

    // address of global variable
    if (auto vexp = e->e1->isVarExp()) {
      LLConstant *c = DtoConstSymbolAddress(e->loc, vexp->var);
      result = c ? DtoBitCast(c, DtoType(e->type)) : nullptr;
      return;
    }

    // address of indexExp
    if (auto iexp = e->e1->isIndexExp()) {
      // indexee must be global static array var
      VarExp *vexp = iexp->e1->isVarExp();
      assert(vexp);
      VarDeclaration *vd = vexp->var->isVarDeclaration();
      assert(vd);
      assert(vd->type->toBasetype()->ty == Tsarray);
      DtoResolveVariable(vd);
      assert(isIrGlobalCreated(vd));

      // get index
      LLConstant *index = toConstElem(iexp->e2);
      assert(index->getType() == DtoSize_t());

      // gep
      LLConstant *idxs[2] = {DtoConstSize_t(0), index};
      LLConstant *val = isaConstant(getIrGlobal(vd)->value);
      val = DtoBitCast(val, DtoType(vd->type->pointerTo()));
      LLConstant *gep = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(val)->getElementType(), val, idxs, true);

      // bitcast to requested type
      assert(e->type->toBasetype()->ty == Tpointer);
      result = DtoBitCast(gep, DtoType(e->type));
      return;
    }

    if (auto se = e->e1->isStructLiteralExp()) {
      result = p->getStructLiteralConstant(se);
      if (result) {
        IF_LOG Logger::cout()
            << "Returning existing global: " << *result << '\n';
        return;
      }

      auto globalVar = new llvm::GlobalVariable(
          p->module, DtoType(se->type), false,
          llvm::GlobalValue::InternalLinkage, nullptr, ".structliteral");
      globalVar->setAlignment(LLMaybeAlign(DtoAlignment(se->type)));

      p->setStructLiteralConstant(se, globalVar);
      llvm::Constant *constValue = toConstElem(se);
      constValue = p->setGlobalVarInitializer(globalVar, constValue, nullptr);
      p->setStructLiteralConstant(se, constValue);

      result = constValue;
      return;
    }

    if (e->e1->op == TOKslice || e->e1->op == TOKdotvar) {
      visit(static_cast<Expression *>(e));
      return;
    }

    llvm_unreachable("unsupported AddrExp in ToConstElemVisitor");
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(FuncExp *e) override {
    IF_LOG Logger::print("FuncExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    FuncLiteralDeclaration *fd = e->fd;
    assert(fd);

    if (fd->tok == TOKreserved && e->type->ty == Tpointer) {
      // This is a lambda that was inferred to be a function literal instead
      // of a delegate, so set tok here in order to get correct types/mangling.
      // Horrible hack, but DMD does the same thing in FuncExp::toElem and
      // other random places.
      fd->tok = TOKfunction;
      fd->vthis = nullptr;
    }

    if (fd->tok != TOKfunction) {
      assert(fd->tok == TOKdelegate || fd->tok == TOKreserved);

      // Only if the function doesn't access its nested context, we can emit a
      // constant delegate with context pointer being null.
      // FIXME: Find a proper way to check whether the context is used.
      //        For now, just enable it for literals declared at module scope.
      if (!fd->toParent2()->isModule()) {
        e->error("non-constant nested delegate literal expression `%s`",
                 e->toChars());
        fatalError(e);
        return;
      }
    }

    // We need to actually codegen the function here, as literals are not
    // added to the module member list.
    Declaration_codegen(fd, p);

    result = DtoCallee(fd, false);
    assert(result);

    if (fd->tok != TOKfunction) {
      auto contextPtr = getNullPtr(getVoidPtrType());
      result = LLConstantStruct::getAnon(gIR->context(), {contextPtr, result});
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ArrayLiteralExp *e) override {
    IF_LOG Logger::print("ArrayLiteralExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type->toChars());
    LOG_SCOPE;

    // extract D types
    Type *bt = e->type->toBasetype();
    Type *elemt = bt->nextOf();

    // build llvm array type
    LLArrayType *arrtype =
        LLArrayType::get(DtoMemType(elemt), e->elements->length);

    // dynamic arrays can occur here as well ...
    bool dyn = (bt->ty != Tsarray);

    llvm::Constant *initval = arrayLiteralToConst(p, e);

    // if static array, we're done
    if (!dyn) {
      result = initval;
      return;
    }

    bool canBeConst = e->type->isConst() || e->type->isImmutable();
    auto gvar = new llvm::GlobalVariable(
        gIR->module, initval->getType(), canBeConst,
        llvm::GlobalValue::InternalLinkage, initval, ".dynarrayStorage");
    gvar->setUnnamedAddr(canBeConst ? llvm::GlobalValue::UnnamedAddr::Global
                                    : llvm::GlobalValue::UnnamedAddr::None);
    llvm::Constant *store = DtoBitCast(gvar, getPtrToType(arrtype));

    if (bt->ty == Tpointer) {
      // we need to return pointer to the static array.
      result = store;
      return;
    }

    // build a constant dynamic array reference with the .ptr field pointing
    // into store
    LLConstant *idxs[2] = {DtoConstUint(0), DtoConstUint(0)};
    LLConstant *globalstorePtr = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(store)->getElementType(), store, idxs, true);

    result = DtoConstSlice(DtoConstSize_t(e->elements->length), globalstorePtr);
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(StructLiteralExp *e) override {
    // type can legitimately be null for ClassReferenceExp::value.
    IF_LOG Logger::print("StructLiteralExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type ? e->type->toChars() : "(null)");
    LOG_SCOPE;

    if (e->useStaticInit) {
      DtoResolveStruct(e->sd);
      result = getIrAggr(e->sd)->getDefaultInit();
    } else {
      // make sure the struct is resolved
      DtoResolveStruct(e->sd);

      std::map<VarDeclaration *, llvm::Constant *> varInits;
      const size_t nexprs = e->elements->length;
      for (size_t i = 0; i < nexprs; i++) {
        if (auto elem = (*e->elements)[i]) {
          LLConstant *c = toConstElem(elem);
          // extend i1 to i8
          if (c->getType() == LLType::getInt1Ty(p->context()))
            c = llvm::ConstantExpr::getZExt(c, LLType::getInt8Ty(p->context()));
          varInits[e->sd->fields[i]] = c;
        }
      }

      result = getIrAggr(e->sd)->createInitializerConstant(varInits);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(ClassReferenceExp *e) override {
    IF_LOG Logger::print("ClassReferenceExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type->toChars());
    LOG_SCOPE;

    ClassDeclaration *origClass = e->originalClass();
    DtoResolveClass(origClass);
    StructLiteralExp *value = e->value;

    result = p->getStructLiteralConstant(value);
    if (result) {
      IF_LOG Logger::cout() << "Using existing global: " << *result << '\n';
    } else {
      auto globalVar = new llvm::GlobalVariable(
          p->module, origClass->type->ctype->isClass()->getMemoryLLType(),
          false, llvm::GlobalValue::InternalLinkage, nullptr, ".classref");
      p->setStructLiteralConstant(value, globalVar);

      std::map<VarDeclaration *, llvm::Constant *> varInits;

      // Unfortunately, ClassReferenceExp::getFieldAt is badly broken – it
      // places the base class fields _after_ those of the subclass.
      {
        const size_t nexprs = value->elements->length;

        std::stack<ClassDeclaration *> classHierachy;
        ClassDeclaration *cur = origClass;
        while (cur) {
          classHierachy.push(cur);
          cur = cur->baseClass;
        }
        size_t i = 0;
        while (!classHierachy.empty()) {
          cur = classHierachy.top();
          classHierachy.pop();
          for (size_t j = 0; j < cur->fields.length; ++j) {
            if (auto elem = (*value->elements)[i]) {
              VarDeclaration *field = cur->fields[j];
              IF_LOG Logger::println("Getting initializer for: %s",
                                     field->toChars());
              LOG_SCOPE;
              varInits[field] = toConstElem(elem);
            }
            ++i;
          }
        }

        (void)nexprs;
        assert(i == nexprs);
      }

      llvm::Constant *constValue =
          getIrAggr(origClass)->createInitializerConstant(varInits);
      constValue = p->setGlobalVarInitializer(globalVar, constValue, nullptr);
      p->setStructLiteralConstant(value, constValue);

      result = constValue;
    }

    if (e->type->ty == Tclass) {
      ClassDeclaration *targetClass = static_cast<TypeClass *>(e->type)->sym;
      if (InterfaceDeclaration *it = targetClass->isInterfaceDeclaration()) {
        assert(it->isBaseOf(origClass, NULL));

        IrTypeClass *typeclass = origClass->type->ctype->isClass();

        // find interface impl
        size_t i_index = typeclass->getInterfaceIndex(it);
        assert(i_index != ~0UL);

        // offset pointer
        result = DtoGEP(result, 0, i_index);
      }
    }

    assert(e->type->ty == Tclass || e->type->ty == Tenum);
    result = DtoBitCast(result, DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(VectorExp *e) override {
    IF_LOG Logger::print("VectorExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    TypeVector *tv = static_cast<TypeVector *>(e->to->toBasetype());
    assert(tv->ty == Tvector);

    const auto elemCount =
        static_cast<TypeSArray *>(tv->basetype)->dim->toInteger();

    // Array literals are assigned element-for-element; other expressions splat
    // across the whole vector.
    if (auto ale = e->e1->isArrayLiteralExp()) {
      llvm::SmallVector<llvm::Constant *, 16> elements;
      elements.reserve(elemCount);
      for (size_t i = 0; i < elemCount; ++i) {
        elements.push_back(toConstElem(indexArrayLiteral(ale, i)));
      }

      result = llvm::ConstantVector::get(elements);
    } else {
      // The AST for
      //   static immutable ubyte16 vec1 = 123;
      // differs from
      //    static immutable ubyte[16] vec1 = 123;
      // In the vector case the AST contains an IntegerExp (of type int) and a
      // CastExp to type ubyte. In the static array case the AST only contains
      // an IntegerExp of type ubyte. Simply call optimize to get rid of the
      // cast.
      // FIXME: Check DMD source to understand why two different ASTs are
      //        constructed.
      result = llvm::ConstantVector::getSplat(
          elemCount, toConstElem(e->e1->optimize(WANTvalue)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(TypeidExp *e) override {
    IF_LOG Logger::print("TypeidExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());

    Type *t = isType(e->obj);
    if (!t) {
      visit(static_cast<Expression *>(e));
      return;
    }

    result = DtoTypeInfoOf(t, /*base=*/false);
    result = DtoBitCast(result, DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////

  void visit(Expression *e) override {
    e->error("expression `%s` is not a constant", e->toChars());
    fatalError(e);
  }
};

LLConstant *toConstElem(Expression *e, IRState *p) {
  return ToConstElemVisitor(p).toConstElem(e);
}
