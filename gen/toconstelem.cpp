//===-- toconstelem.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typeinf.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include "template.h"
// Needs other includes.
#include "ctfe.h"

extern dinteger_t undoStrideMul(Loc &loc, Type *t, dinteger_t offset);

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

  //////////////////////////////////////////////////////////////////////////////////////////

  LLConstant *toConstElem(Expression *e) {
    result = nullptr;
    e->accept(this);
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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
      LLType *vartype = DtoType(e->type);
      result = DtoTypeInfoOf(ti->tinfo, false);
      if (result->getType() != getPtrToType(vartype)) {
        result = llvm::ConstantExpr::getBitCast(result, vartype);
      }
      return;
    }

    VarDeclaration *vd = e->var->isVarDeclaration();
    if (vd && vd->isConst() && vd->init) {
      if (vd->inuse) {
        e->error("recursive reference %s", e->toChars());
        result = llvm::UndefValue::get(DtoType(e->type));
      } else {
        vd->inuse++;
        // return the initializer
        result = DtoConstInitializer(e->loc, e->type, vd->init);
        vd->inuse--;
      }
    }
    // fail
    else {
      e->error("non-constant expression %s", e->toChars());
      result = llvm::UndefValue::get(DtoType(e->type));
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(RealExp *e) override {
    IF_LOG Logger::print("RealExp::toConstElem: %s @ %s | %La\n", e->toChars(),
                         e->type->toChars(), e->value);
    LOG_SCOPE;
    Type *t = e->type->toBasetype();
    result = DtoConstFP(t, e->value);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(ComplexExp *e) override {
    IF_LOG Logger::print("ComplexExp::toConstElem(): %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;
    result = DtoConstComplex(e->type, e->value.re, e->value.im);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(StringExp *e) override {
    IF_LOG Logger::print("StringExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    Type *t = e->type->toBasetype();
    Type *cty = t->nextOf()->toBasetype();

    bool nullterm = (t->ty != Tsarray);
    size_t endlen = nullterm ? e->len + 1 : e->len;

    LLType *ct = DtoMemType(cty);
    LLArrayType *at = LLArrayType::get(ct, endlen);

    llvm::StringMap<llvm::GlobalVariable *> *stringLiteralCache = nullptr;
    LLConstant *_init;
    switch (cty->size()) {
    default:
      llvm_unreachable("Unknown char type");
    case 1:
      _init = toConstantArray(ct, at, static_cast<uint8_t *>(e->string), e->len,
                              nullterm);
      stringLiteralCache = &(gIR->stringLiteral1ByteCache);
      break;
    case 2:
      _init = toConstantArray(ct, at, static_cast<uint16_t *>(e->string),
                              e->len, nullterm);
      stringLiteralCache = &(gIR->stringLiteral2ByteCache);
      break;
    case 4:
      _init = toConstantArray(ct, at, static_cast<uint32_t *>(e->string),
                              e->len, nullterm);
      stringLiteralCache = &(gIR->stringLiteral4ByteCache);
      break;
    }

    if (t->ty == Tsarray) {
      result = _init;
      return;
    }

    llvm::StringRef key(e->toChars());
    llvm::GlobalVariable *gvar =
        (stringLiteralCache->find(key) == stringLiteralCache->end())
            ? nullptr
            : (*stringLiteralCache)[key];
    if (gvar == nullptr) {
      llvm::GlobalValue::LinkageTypes _linkage =
          llvm::GlobalValue::PrivateLinkage;
      gvar = new llvm::GlobalVariable(gIR->module, _init->getType(), true,
                                      _linkage, _init, ".str");
      gvar->setUnnamedAddr(true);
      (*stringLiteralCache)[key] = gvar;
    }

    llvm::ConstantInt *zero =
        LLConstantInt::get(LLType::getInt32Ty(gIR->context()), 0, false);
    LLConstant *idxs[2] = {zero, zero};
#if LDC_LLVM_VER >= 307
    LLConstant *arrptr = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(gvar)->getElementType(), gvar, idxs, true);
#else
    LLConstant *arrptr = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true);
#endif

    if (t->ty == Tpointer) {
      result = arrptr;
    } else if (t->ty == Tarray) {
      LLConstant *clen = LLConstantInt::get(DtoSize_t(), e->len, false);
      result = DtoConstSlice(clen, arrptr, e->type);
    } else {
      llvm_unreachable("Unknown type for StringExp.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(AddExp *e) override {
    IF_LOG Logger::print("AddExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    // add to pointer
    Type *t1b = e->e1->type->toBasetype();
    if (t1b->ty == Tpointer && e->e2->type->isintegral()) {
      llvm::Constant *ptr = toConstElem(e->e1);
      dinteger_t idx = undoStrideMul(e->loc, t1b, e->e2->toInteger());
#if LDC_LLVM_VER >= 307
      result = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(ptr)->getElementType(), ptr, DtoConstSize_t(idx));
#else
      result = llvm::ConstantExpr::getGetElementPtr(ptr, DtoConstSize_t(idx));
#endif
    } else {
      e->error("expression '%s' is not a constant", e->toChars());
      if (!global.gag) {
        fatal();
      }
      result = llvm::UndefValue::get(DtoType(e->type));
    }
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
#if LDC_LLVM_VER >= 307
      result = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(ptr)->getElementType(), ptr, negIdx);
#else
      result = llvm::ConstantExpr::getGetElementPtr(ptr, negIdx);
#endif
    } else {
      e->error("expression '%s' is not a constant", e->toChars());
      if (!global.gag) {
        fatal();
      }
      result = llvm::UndefValue::get(DtoType(e->type));
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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
      e->error("ct cast of string to dynamic array not fully implemented");
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
#if LDC_LLVM_VER >= 307
        value = llvm::ConstantExpr::getGetElementPtr(
            isaPointer(value)->getElementType(), value, idxs, true);
#else
        value = llvm::ConstantExpr::getGetElementPtr(value, idxs, true);
#endif
      }
      result = DtoBitCast(value, DtoType(tb));
    } else if (tb->ty == Tclass && e->e1->type->ty == Tclass) {
      assert(e->e1->op == TOKclassreference);
      ClassDeclaration *cd =
          static_cast<ClassReferenceExp *>(e->e1)->originalClass();

      llvm::Constant *instance = toConstElem(e->e1);
      if (InterfaceDeclaration *it =
              static_cast<TypeClass *>(tb)->sym->isInterfaceDeclaration()) {
        assert(it->isBaseOf(cd, NULL));

        IrTypeClass *typeclass = cd->type->ctype->isClass();

        // find interface impl
        size_t i_index = typeclass->getInterfaceIndex(it);
        assert(i_index != ~0UL);

        // offset pointer
        instance = DtoGEPi(instance, 0, i_index);
      }
      result = DtoBitCast(instance, DtoType(tb));
    } else {
      goto Lerr;
    }
    return;

  Lerr:
    e->error("cannot cast %s to %s at compile time", e->e1->type->toChars(),
             e->type->toChars());
    if (!global.gag) {
      fatal();
    }
    result = llvm::UndefValue::get(DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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
        result = llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
            NULL,
#endif
            base, DtoConstSize_t(e->offset / elemSize));
      } else {
        // Offset isn't a multiple of base type size, just cast to i8* and
        // apply the byte offset.
        result = llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
            NULL,
#endif
            DtoBitCast(base, getVoidPtrType()), DtoConstSize_t(e->offset));
      }
    }

    result = DtoBitCast(result, DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(AddrExp *e) override {
    IF_LOG Logger::println("AddrExp::toConstElem: %s @ %s", e->toChars(),
                           e->type->toChars());
    LOG_SCOPE;
    // FIXME: this should probably be generalized more so we don't
    // need to have a case for each thing we can take the address of

    // address of global variable
    if (e->e1->op == TOKvar) {
      VarExp *vexp = static_cast<VarExp *>(e->e1);
      LLConstant *c = DtoConstSymbolAddress(e->loc, vexp->var);
      result = c ? DtoBitCast(c, DtoType(e->type)) : nullptr;
    }
    // address of indexExp
    else if (e->e1->op == TOKindex) {
      IndexExp *iexp = static_cast<IndexExp *>(e->e1);

      // indexee must be global static array var
      assert(iexp->e1->op == TOKvar);
      VarExp *vexp = static_cast<VarExp *>(iexp->e1);
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
#if LDC_LLVM_VER >= 307
      LLConstant *gep = llvm::ConstantExpr::getGetElementPtr(
          isaPointer(val)->getElementType(), val, idxs, true);
#else
      LLConstant *gep = llvm::ConstantExpr::getGetElementPtr(val, idxs, true);
#endif

      // bitcast to requested type
      assert(e->type->toBasetype()->ty == Tpointer);
      result = DtoBitCast(gep, DtoType(e->type));
    } else if (e->e1->op == TOKstructliteral) {
      StructLiteralExp *se = static_cast<StructLiteralExp *>(e->e1);

      if (se->globalVar) {
        IF_LOG Logger::cout() << "Returning existing global: " << *se->globalVar
                              << '\n';
        result = se->globalVar;
        return;
      }

      se->globalVar = new llvm::GlobalVariable(
          p->module, DtoType(e->e1->type), false,
          llvm::GlobalValue::InternalLinkage, nullptr, ".structliteral");

      llvm::Constant *constValue = toConstElem(se);
      if (constValue->getType() !=
          se->globalVar->getType()->getContainedType(0)) {
        auto finalGlobalVar = new llvm::GlobalVariable(
            p->module, constValue->getType(), false,
            llvm::GlobalValue::InternalLinkage, nullptr, ".structliteral");
        se->globalVar->replaceAllUsesWith(
            DtoBitCast(finalGlobalVar, se->globalVar->getType()));
        se->globalVar->eraseFromParent();
        se->globalVar = finalGlobalVar;
      }
      se->globalVar->setInitializer(constValue);
      se->globalVar->setAlignment(DtoAlignment(se->type));

      result = se->globalVar;
    } else if (e->e1->op == TOKslice) {
      e->error("non-constant expression '%s'", e->toChars());
      if (!global.gag) {
        fatal();
      }
      result = llvm::UndefValue::get(DtoType(e->type));
    }
    // not yet supported
    else {
      e->error("constant expression '%s' not yet implemented", e->toChars());
      fatal();
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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
      e->error("non-constant nested delegate literal expression %s",
               e->toChars());
      if (!global.gag) {
        fatal();
      }
      result = llvm::UndefValue::get(DtoType(e->type));
    } else {
      // We need to actually codegen the function here, as literals are not
      // added
      // to the module member list.
      Declaration_codegen(fd, p);
      assert(getIrFunc(fd)->func);

      result = getIrFunc(fd)->func;
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(ArrayLiteralExp *e) override {
    IF_LOG Logger::print("ArrayLiteralExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type->toChars());
    LOG_SCOPE;

    // extract D types
    Type *bt = e->type->toBasetype();
    Type *elemt = bt->nextOf();

    // build llvm array type
    LLArrayType *arrtype =
        LLArrayType::get(DtoMemType(elemt), e->elements->dim);

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
    gvar->setUnnamedAddr(canBeConst);
    llvm::Constant *store = DtoBitCast(gvar, getPtrToType(arrtype));

    if (bt->ty == Tpointer) {
      // we need to return pointer to the static array.
      result = store;
      return;
    }

    // build a constant dynamic array reference with the .ptr field pointing
    // into store
    LLConstant *idxs[2] = {DtoConstUint(0), DtoConstUint(0)};
#if LDC_LLVM_VER >= 307
    LLConstant *globalstorePtr = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(store)->getElementType(), store, idxs, true);
#else
    LLConstant *globalstorePtr =
        llvm::ConstantExpr::getGetElementPtr(store, idxs, true);
#endif

    result = DtoConstSlice(DtoConstSize_t(e->elements->dim), globalstorePtr);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(StructLiteralExp *e) override {
    // type can legitimately be null for ClassReferenceExp::value.
    IF_LOG Logger::print("StructLiteralExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type ? e->type->toChars() : "(null)");
    LOG_SCOPE;

    if (e->sinit) {
      // Copied from VarExp::toConstElem, need to clean this mess up.
      Type *sdecltype = e->sinit->type->toBasetype();
      IF_LOG Logger::print("Sym: type=%s\n", sdecltype->toChars());
      assert(sdecltype->ty == Tstruct);
      TypeStruct *ts = static_cast<TypeStruct *>(sdecltype);
      DtoResolveStruct(ts->sym);

      result = getIrAggr(ts->sym)->getDefaultInit();
    } else {
      // make sure the struct is resolved
      DtoResolveStruct(e->sd);

      std::map<VarDeclaration *, llvm::Constant *> varInits;
      const size_t nexprs = e->elements->dim;
      for (size_t i = 0; i < nexprs; i++) {
        if ((*e->elements)[i]) {
          varInits[e->sd->fields[i]] = toConstElem((*e->elements)[i]);
        }
      }

      result = getIrAggr(e->sd)->createInitializerConstant(varInits);
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(ClassReferenceExp *e) override {
    IF_LOG Logger::print("ClassReferenceExp::toConstElem: %s @ %s\n",
                         e->toChars(), e->type->toChars());
    LOG_SCOPE;

    ClassDeclaration *origClass = e->originalClass();
    DtoResolveClass(origClass);
    StructLiteralExp *value = e->value;

    if (value->globalVar) {
      IF_LOG Logger::cout() << "Using existing global: " << *value->globalVar
                            << '\n';
    } else {
      value->globalVar = new llvm::GlobalVariable(
          p->module, origClass->type->ctype->isClass()->getMemoryLLType(),
          false, llvm::GlobalValue::InternalLinkage, nullptr, ".classref");

      std::map<VarDeclaration *, llvm::Constant *> varInits;

      // Unfortunately, ClassReferenceExp::getFieldAt is badly broken – it
      // places the base class fields _after_ those of the subclass.
      {
        const size_t nexprs = value->elements->dim;

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
          for (size_t j = 0; j < cur->fields.dim; ++j) {
            if ((*value->elements)[i]) {
              VarDeclaration *field = cur->fields[j];
              IF_LOG Logger::println("Getting initializer for: %s",
                                     field->toChars());
              LOG_SCOPE;
              varInits[field] = toConstElem((*value->elements)[i]);
            }
            ++i;
          }
        }
        assert(i == nexprs);
      }

      llvm::Constant *constValue =
          getIrAggr(origClass)->createInitializerConstant(varInits);

      if (constValue->getType() !=
          value->globalVar->getType()->getContainedType(0)) {
        auto finalGlobalVar = new llvm::GlobalVariable(
            p->module, constValue->getType(), false,
            llvm::GlobalValue::InternalLinkage, nullptr, ".classref");
        value->globalVar->replaceAllUsesWith(
            DtoBitCast(finalGlobalVar, value->globalVar->getType()));
        value->globalVar->eraseFromParent();
        value->globalVar = finalGlobalVar;
      }
      value->globalVar->setInitializer(constValue);
    }

    result = value->globalVar;

    if (e->type->ty == Tclass) {
      ClassDeclaration *targetClass = static_cast<TypeClass *>(e->type)->sym;
      if (InterfaceDeclaration *it = targetClass->isInterfaceDeclaration()) {
        assert(it->isBaseOf(origClass, NULL));

        IrTypeClass *typeclass = origClass->type->ctype->isClass();

        // find interface impl
        size_t i_index = typeclass->getInterfaceIndex(it);
        assert(i_index != ~0UL);

        // offset pointer
        result = DtoGEPi(result, 0, i_index);
      }
    }

    assert(e->type->ty == Tclass || e->type->ty == Tenum);
    result = DtoBitCast(result, DtoType(e->type));
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(VectorExp *e) override {
    IF_LOG Logger::print("VectorExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());
    LOG_SCOPE;

    TypeVector *tv = static_cast<TypeVector *>(e->to->toBasetype());
    assert(tv->ty == Tvector);

    // The AST for
    //   static immutable ubyte16 vec1 = 123;
    // differs from
    //    static immutable ubyte[16] vec1 = 123;
    // In the vector case the AST contains an IntegerExp (of type int) and a
    // CastExp to type ubyte. In the static array case the AST only contains an
    // IntegerExp of type ubyte. Simply call optimize to get  rid of the cast.
    // FIXME: Check DMD source to understand why two different ASTs are
    //        constructed.
    llvm::Constant *val = toConstElem(e->e1->optimize(WANTvalue));

    dinteger_t elemCount =
        static_cast<TypeSArray *>(tv->basetype)->dim->toInteger();
    result = llvm::ConstantVector::getSplat(elemCount, val);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(TypeidExp *e) override {
    IF_LOG Logger::print("TypeidExp::toConstElem: %s @ %s\n", e->toChars(),
                         e->type->toChars());

    Type *t = isType(e->obj);
    if (!t) {
      visit(static_cast<Expression *>(e));
      return;
    }

    TypeInfoDeclaration *tid = getOrCreateTypeInfoDeclaration(t, nullptr);
    TypeInfoDeclaration_codegen(tid, p);
    result = llvm::cast<llvm::GlobalVariable>(getIrGlobal(tid)->value);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  void visit(Expression *e) override {
    e->error("expression '%s' is not a constant", e->toChars());
    if (!global.gag) {
      fatal();
    }

    // Do not return null here, as AssocArrayLiteralExp::toElem determines
    // whether it can allocate the needed arrays statically by just invoking
    // toConstElem on its key/value expressions, and handling the null value
    // consequently would require error-prone adaptions in all other code.
    result = llvm::UndefValue::get(DtoType(e->type));
  }
};

LLConstant *toConstElem(Expression *e, IRState *p) {
  return ToConstElemVisitor(p).toConstElem(e);
}
