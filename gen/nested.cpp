//===-- nested.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/nested.h"

#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/target.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irtypeaggr.h"
#include "llvm/Analysis/ValueTracking.h"

static unsigned getVthisIdx(AggregateDeclaration *ad) {
  return getFieldGEPIndex(ad, ad->vthis);
}

static void DtoCreateNestedContextType(FuncDeclaration *fd);

DValue *DtoNestedVariable(Loc &loc, Type *astype, VarDeclaration *vd,
                          bool byref) {
  IF_LOG Logger::println("DtoNestedVariable for %s @ %s", vd->toChars(),
                         loc.toChars());
  LOG_SCOPE;

  ////////////////////////////////////
  // Locate context value

  Dsymbol *vdparent = vd->toParent2();
  assert(vdparent);

  IrFunction *irfunc = gIR->func();

  // Check whether we can access the needed frame
  FuncDeclaration *fd = irfunc->decl;
  while (fd && fd != vdparent) {
    fd = getParentFunc(fd);
  }
  if (!fd) {
    error(loc, "function `%s` cannot access frame of function `%s`",
          irfunc->decl->toPrettyChars(), vdparent->toPrettyChars());
    return new DLValue(astype, llvm::UndefValue::get(DtoPtrToType(astype)));
  }

  // is the nested variable in this scope?
  if (vdparent == irfunc->decl) {
    return makeVarDValue(astype, vd);
  }

  // get the nested context
  LLValue *ctx = nullptr;
  bool skipDIDeclaration = false;
  auto currentCtx = gIR->funcGen().nestedVar;
  if (currentCtx) {
    Logger::println("Using own nested context of current function");
    ctx = currentCtx;
  } else if (irfunc->decl->isMember2()) {
    Logger::println(
        "Current function is member of nested class, loading vthis");

    AggregateDeclaration *cd = irfunc->decl->isMember2();
    LLValue *val = irfunc->thisArg;
    if (cd->isClassDeclaration()) {
      val = DtoLoad(val);
    }
    ctx = DtoLoad(DtoGEP(val, 0, getVthisIdx(cd), ".vthis"));
    skipDIDeclaration = true;
  } else {
    Logger::println("Regular nested function, using context arg");
    ctx = irfunc->nestArg;
  }

  assert(ctx);
  IF_LOG { Logger::cout() << "Context: " << *ctx << '\n'; }

  DtoCreateNestedContextType(vdparent->isFuncDeclaration());

  assert(isIrLocalCreated(vd));
  IrLocal *const irLocal = getIrLocal(vd);

  // The variable may not actually be nested in a speculative context (e.g.,
  // with `-allinst`, see https://github.com/ldc-developers/ldc/issues/2932).
  // Use an invalid null storage in that case, so that accessing the var at
  // runtime will cause a segfault.
  if (irLocal->nestedIndex == -1) {
    Logger::println(
        "WARNING: isn't actually nested, using invalid null storage");
    auto llType = DtoPtrToType(astype);
    if (isSpecialRefVar(vd))
      llType = llType->getPointerTo();
    return makeVarDValue(astype, vd, llvm::ConstantPointerNull::get(llType));
  }

  ////////////////////////////////////
  // Extract variable from nested context

  assert(irfunc->frameType);
  const auto frameType = LLPointerType::getUnqual(irfunc->frameType);
  IF_LOG { Logger::cout() << "casting to: " << *irfunc->frameType << '\n'; }
  LLValue *val = DtoBitCast(ctx, frameType);

  // Make the DWARF variable address relative to the context pointer (ctx);
  // register all ops (offsetting, dereferencing) required to get there in the
  // following list.
  LLSmallVector<int64_t, 4> dwarfAddrOps;

  const auto offsetToNthField = [&val, &dwarfAddrOps](unsigned fieldIndex,
                                                      const char *name = "") {
    gIR->DBuilder.OpOffset(dwarfAddrOps, val, fieldIndex);
    val = DtoGEP(val, 0, fieldIndex, name);
  };
  const auto dereference = [&val, &dwarfAddrOps](const char *name = "") {
    gIR->DBuilder.OpDeref(dwarfAddrOps);
    val = DtoAlignedLoad(val, name);
  };

  const auto vardepth = irLocal->nestedDepth;
  const auto funcdepth = irfunc->depth;

  IF_LOG {
    Logger::cout() << "Variable: " << vd->toChars() << '\n';
    Logger::cout() << "Variable depth: " << vardepth << '\n';
    Logger::cout() << "Function: " << irfunc->decl->toChars() << '\n';
    Logger::cout() << "Function depth: " << funcdepth << '\n';
  }

  if (vardepth == funcdepth) {
    // This is not always handled above because functions without
    // variables accessed by nested functions don't create new frames.
    IF_LOG Logger::println("Same depth");
  } else {
    // Load frame pointer and index that...
    IF_LOG Logger::println("Lower depth");
    offsetToNthField(vardepth);
    IF_LOG Logger::cout() << "Frame index: " << *val << '\n';
    dereference((std::string(".frame.") + vdparent->toChars()).c_str());
    IF_LOG Logger::cout() << "Frame: " << *val << '\n';
  }

  offsetToNthField(irLocal->nestedIndex, vd->toChars());
  IF_LOG {
    Logger::cout() << "Addr: " << *val << '\n';
    Logger::cout() << "of type: " << *val->getType() << '\n';
  }
  const bool isRefOrOut = vd->isRef() || vd->isOut();
  if (isSpecialRefVar(vd)) {
    // Handled appropriately by makeVarDValue() and EmitLocalVariable(), pass
    // storage of pointer (reference lvalue).
  } else if (byref || isRefOrOut) {
    val = DtoAlignedLoad(val);
    // ref/out variables get a reference-debuginfo-type in EmitLocalVariable()
    // => don't dereference, use reference lvalue as address
    if (!isRefOrOut)
      gIR->DBuilder.OpDeref(dwarfAddrOps);
    IF_LOG {
      Logger::cout() << "Was byref, now: " << *irLocal->value << '\n';
      Logger::cout() << "of type: " << *irLocal->value->getType() << '\n';
    }
  }

  if (!skipDIDeclaration && global.params.symdebug) {
#if LDC_LLVM_VER < 500
    gIR->DBuilder.OpDeref(dwarfAddrOps);
#endif
    gIR->DBuilder.EmitLocalVariable(ctx, vd, nullptr, false,
                                    /*forceAsLocal=*/true, false, dwarfAddrOps);
  }

  return makeVarDValue(astype, vd, val);
}

void DtoResolveNestedContext(Loc &loc, AggregateDeclaration *decl,
                             LLValue *value) {
  IF_LOG Logger::println("Resolving nested context");
  LOG_SCOPE;

  // get context
  LLValue *nest = DtoNestedContext(loc, decl);

  // store into right location
  if (!llvm::dyn_cast<llvm::UndefValue>(nest)) {
    // Need to make sure the declaration has already been resolved, because
    // when multiple source files are specified on the command line, the
    // frontend sometimes adds "nested" (i.e. a template in module B
    // instantiated from module A with a type from module A instantiates
    // another template from module B) into the wrong module, messing up
    // our codegen order.
    DtoResolveDsymbol(decl);

    unsigned idx = getVthisIdx(decl);
    LLValue *gep = DtoGEP(value, 0, idx, ".vthis");
    DtoStore(DtoBitCast(nest, gep->getType()->getContainedType(0)), gep);
  }
}

LLValue *DtoNestedContext(Loc &loc, Dsymbol *sym) {
  IF_LOG Logger::println("DtoNestedContext for %s", sym->toPrettyChars());
  LOG_SCOPE;

  // Exit quickly for functions that accept a context pointer for ABI purposes,
  // but do not actually read from it.
  //
  // null is used instead of LLVM's undef to not break bitwise comparison,
  // for instances of nested struct types which don't have any nested
  // references, or for delegates to nested functions with an empty context.
  //
  // We cannot simply fall back to retuning null once we discover that we
  // don't actually have a context to pass, because we sadly also need to
  // catch invalid code here in the glue layer (see error() below).
  if (FuncDeclaration *symfd = sym->isFuncDeclaration()) {
    // Make sure we've had a chance to analyze nested context usage
    DtoCreateNestedContextType(symfd);

    int depth = getIrFunc(symfd)->depth;
    Logger::println("for function of depth %d", depth);
    if (depth == -1 || (depth == 0 && !symfd->closureVars.empty())) {
      Logger::println("function does not have context or creates its own "
                      "from scratch, returning null");
      return llvm::ConstantPointerNull::get(getVoidPtrType());
    }
  }

  // The function we are currently in, and the constructed object/called
  // function might inherit a context pointer from.
  auto &funcGen = gIR->funcGen();
  auto &irFunc = funcGen.irFunc;

  bool fromParent = true;

  LLValue *val;
  if (funcGen.nestedVar) {
    // if this func has its own vars that are accessed by nested funcs
    // use its own context
    val = funcGen.nestedVar;
    fromParent = false;
  } else if (irFunc.nestArg) {
    // otherwise, it may have gotten a context from the caller
    val = irFunc.nestArg;
  } else if (irFunc.thisArg) {
    // or just have a this argument
    AggregateDeclaration *ad = irFunc.decl->isMember2();
    val = ad->isClassDeclaration() ? DtoLoad(irFunc.thisArg) : irFunc.thisArg;
    if (!ad->vthis) {
      // This is just a plain 'outer' reference of a class nested in a
      // function (but without any variables in the nested context).
      return val;
    }
    val = DtoLoad(DtoGEP(val, 0, getVthisIdx(ad), ".vthis"));
  } else {
    if (sym->isFuncDeclaration()) {
      // If we are here, the function actually needs its nested context
      // and we cannot provide one. Thus, it's invalid code that is
      // unfortunately not caught in the frontend (e.g. a function literal
      // tries to call a nested function from the parent scope).
      error(
          loc,
          "function `%s` is a nested function and cannot be accessed from `%s`",
          sym->toPrettyChars(), irFunc.decl->toPrettyChars());
      fatal();
    }
    return llvm::ConstantPointerNull::get(getVoidPtrType());
  }

  // The symbol may need a parent context of the current function.
  if (FuncDeclaration *frameToPass = getParentFunc(sym)) {
    IF_LOG Logger::println("Parent frame is from %s", frameToPass->toChars());
    FuncDeclaration *ctxfd = irFunc.decl;
    IF_LOG Logger::println("Current function is %s", ctxfd->toChars());
    if (fromParent) {
      ctxfd = getParentFunc(ctxfd);
      assert(ctxfd && "Context from outer function, but no outer function?");
    }
    IF_LOG Logger::println("Context is from %s", ctxfd->toChars());

    unsigned neededDepth = getIrFunc(frameToPass)->depth;
    unsigned ctxDepth = getIrFunc(ctxfd)->depth;

    IF_LOG {
      Logger::cout() << "Needed depth: " << neededDepth << '\n';
      Logger::cout() << "Context depth: " << ctxDepth << '\n';
    }

    if (neededDepth >= ctxDepth) {
      // assert(neededDepth <= ctxDepth + 1 && "How are we going more than one
      // nesting level up?");
      // fd needs the same context as we do, so all is well
      IF_LOG Logger::println(
          "Calling sibling function or directly nested function");
    } else {
      val = DtoBitCast(val,
                       LLPointerType::getUnqual(getIrFunc(ctxfd)->frameType));
      val = DtoGEP(val, 0, neededDepth);
      val = DtoAlignedLoad(
          val, (std::string(".frame.") + frameToPass->toChars()).c_str());
    }
  }

  IF_LOG {
    Logger::cout() << "result = " << *val << '\n';
    Logger::cout() << "of type " << *val->getType() << '\n';
  }
  return val;
}

static void DtoCreateNestedContextType(FuncDeclaration *fd) {
  IF_LOG Logger::println("DtoCreateNestedContextType for %s",
                         fd->toPrettyChars());
  LOG_SCOPE

  DtoDeclareFunction(fd);

  IrFunction &irFunc = *getIrFunc(fd);

  if (irFunc.nestedContextCreated) {
    Logger::println("already done");
    return;
  }
  irFunc.nestedContextCreated = true;

  FuncDeclaration *parentFunc = getParentFunc(fd);
  // Make sure the parent has already been analyzed.
  if (parentFunc) {
    DtoCreateNestedContextType(parentFunc);
  }

  if (fd->closureVars.length == 0) {
    // No local variables of this function are captured.
    if (parentFunc) {
      // Propagate context arg properties if the context arg is passed on
      // unmodified.
      IrFunction &parentIrFunc = *getIrFunc(parentFunc);
      irFunc.frameType = parentIrFunc.frameType;
      irFunc.frameTypeAlignment = parentIrFunc.frameTypeAlignment;
      irFunc.depth = parentIrFunc.depth;
    }
    return;
  }

  Logger::println("has nested frame");

  // construct nested variables array
  // start with adding all enclosing parent frames until a static parent is
  // reached

  LLStructType *innerFrameType = nullptr;
  unsigned depth = 0;

  if (parentFunc) {
    IrFunction &parentIrFunc = *getIrFunc(parentFunc);
    innerFrameType = parentIrFunc.frameType;
    if (innerFrameType) {
      depth = parentIrFunc.depth + 1;
    }
  }

  irFunc.depth = depth;

  IF_LOG Logger::cout() << "Function " << fd->toChars() << " has depth "
                        << depth << '\n';

  AggrTypeBuilder builder(false);

  if (depth != 0) {
    assert(innerFrameType);
    unsigned ptrSize = gDataLayout->getPointerSize();
    // Add frame pointer types for all but last frame
    for (unsigned i = 0; i < (depth - 1); ++i) {
      builder.addType(innerFrameType->getElementType(i), ptrSize);
    }
    // Add frame pointer type for last frame
    builder.addType(LLPointerType::getUnqual(innerFrameType), ptrSize);
  }

  // Add the direct nested variables of this function, and update their
  // indices to match.
  // TODO: optimize ordering for minimal space usage?
  for (auto vd : fd->closureVars) {
    unsigned alignment = DtoAlignment(vd);
    if (alignment > 1) {
      builder.alignCurrentOffset(alignment);
    }

    IrLocal &irLocal = *getIrLocal(vd, true);
    irLocal.nestedIndex = builder.currentFieldIndex();
    irLocal.nestedDepth = depth;

    LLType *t = nullptr;
    if (vd->isRef() || vd->isOut()) {
      t = DtoType(vd->type->pointerTo());
    } else if (vd->isParameter() && (vd->storage_class & STClazy)) {
      // The LL type is a delegate (LL struct).
      Type *dt = TypeFunction::create(nullptr, vd->type, VARARGnone, LINKd);
      dt = createTypeDelegate(dt);
      dt = merge(dt);
      t = DtoType(dt);
    } else {
      t = DtoMemType(vd->type);
    }

    builder.addType(t, getTypeAllocSize(t));

    IF_LOG Logger::cout() << "Nested var '" << vd->toChars() << "' of type "
                          << *t << "\n";
  }

  LLStructType *frameType =
      LLStructType::create(gIR->context(), builder.defaultTypes(),
                           std::string("nest.") + fd->toChars());

  IF_LOG Logger::cout() << "frameType = " << *frameType << '\n';

  // Store type in IrFunction
  irFunc.frameType = frameType;
  irFunc.frameTypeAlignment = builder.overallAlignment();
}

void DtoCreateNestedContext(FuncGenState &funcGen) {
  const auto fd = funcGen.irFunc.decl;
  IF_LOG Logger::println("DtoCreateNestedContext for %s", fd->toPrettyChars());
  LOG_SCOPE

  DtoCreateNestedContextType(fd);

  // construct nested variables array
  if (fd->closureVars.length > 0) {
    auto &irFunc = funcGen.irFunc;
    unsigned depth = irFunc.depth;
    LLStructType *frameType = irFunc.frameType;
    // Create frame for current function and append to frames list
    LLValue *frame = nullptr;
    bool needsClosure = fd->needsClosure();
    IF_LOG Logger::println("Needs closure (GC) flag: %d", (int)needsClosure);
    if (needsClosure) {
      // FIXME: alignment ?
      frame = DtoGcMalloc(fd->loc, frameType, ".frame");
    } else {
      unsigned alignment =
          std::max(getABITypeAlign(frameType), irFunc.frameTypeAlignment);
      frame = DtoRawAlloca(frameType, alignment, ".frame");
    }

    // copy parent frames into beginning
    if (depth != 0) {
      LLValue *src = irFunc.nestArg;
      if (!src) {
        assert(irFunc.thisArg);
        assert(fd->isMember2());
        LLValue *thisval = DtoLoad(irFunc.thisArg);
        AggregateDeclaration *cd = fd->isMember2();
        assert(cd);
        assert(cd->vthis);
        IF_LOG Logger::println("Indexing to 'this'");
        if (cd->isStructDeclaration()) {
          src = DtoExtractValue(thisval, getVthisIdx(cd), ".vthis");
        } else {
          src = DtoLoad(DtoGEP(thisval, 0, getVthisIdx(cd), ".vthis"));
        }
      }
      if (depth > 1) {
        src = DtoBitCast(src, getVoidPtrType());
        LLValue *dst = DtoBitCast(frame, getVoidPtrType());
        DtoMemCpy(dst, src, DtoConstSize_t((depth - 1) * target.ptrsize),
                  getABITypeAlign(getVoidPtrType()));
      }
      // Copy nestArg into framelist; the outer frame is not in the list of
      // pointers
      src = DtoBitCast(src, frameType->getContainedType(depth - 1));
      LLValue *gep = DtoGEP(frame, 0, depth - 1);
      DtoAlignedStore(src, gep);
    }

    funcGen.nestedVar = frame;

    // go through all nested vars and assign addresses where possible.
    for (auto vd : fd->closureVars) {
      if (needsClosure && vd->needsScopeDtor()) {
        // This should really be a front-end, not a glue layer error,
        // but we need to fix this in DMD too.
        vd->error("has scoped destruction, cannot build closure");
      }

      IrLocal *irLocal = getIrLocal(vd);
      LLValue *gep = DtoGEP(frame, 0, irLocal->nestedIndex, vd->toChars());
      if (vd->isParameter()) {
        IF_LOG Logger::println("nested param: %s", vd->toChars());
        LOG_SCOPE
        IrParameter *parm = getIrParameter(vd);
        assert(parm->value);
        assert(parm->value->getType()->isPointerTy());

        if (vd->isRef() || vd->isOut()) {
          Logger::println(
              "Captured by reference, copying pointer to nested frame");
          DtoAlignedStore(parm->value, gep);
          // pass GEP as reference lvalue to EmitLocalVariable()
        } else {
          Logger::println("Copying to nested frame");
          // The parameter value is an alloca'd stack slot.
          // Copy to the nesting frame and leave the alloca for
          // the optimizers to clean up.
          DtoMemCpy(gep, parm->value);
          gep->takeName(parm->value);
          parm->value = gep;
        }
      } else {
        IF_LOG Logger::println("nested var:   %s", vd->toChars());
        assert(!irLocal->value);
        irLocal->value = gep;
      }

      if (global.params.symdebug) {
        LLSmallVector<int64_t, 1> dwarfAddrOps;
#if LDC_LLVM_VER < 500
        // Because we are passing a GEP instead of an alloca to
        // llvm.dbg.declare, we have to make the address dereference explicit.
        gIR->DBuilder.OpDeref(dwarfAddrOps);
#endif
        gIR->DBuilder.EmitLocalVariable(gep, vd, nullptr, false, false, false,
                                        dwarfAddrOps);
      }
    }
  }
}
