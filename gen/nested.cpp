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
#include "dmd/target.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irtypeaggr.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace dmd;

namespace {
unsigned getVthisIdx(AggregateDeclaration *ad) {
  return getFieldGEPIndex(ad, ad->vthis);
}

bool isNRVOVar(VarDeclaration *vd) {
  if (auto fd = vd->toParent2()->isFuncDeclaration())
    return fd->isNRVO() && vd == fd->nrvo_var && !fd->needsClosure();
  return false;
}

bool captureByRef(VarDeclaration *vd) {
  return vd->isReference() || isNRVOVar(vd);
}
LLValue *loadThisPtr(AggregateDeclaration *ad, IrFunction &irfunc) {
  if (ad->isClassDeclaration()) {
      return DtoLoad(DtoType(irfunc.irFty.arg_this->type),
                     irfunc.thisArg);
  }

  return irfunc.thisArg;
}

LLValue *indexVThis(AggregateDeclaration *ad,  LLValue* val) {
  DtoResolveDsymbol(ad);
  llvm::StructType *st = getIrAggr(ad)->getLLStructType();
  unsigned idx = getVthisIdx(ad);
  return DtoLoad(st->getElementType(idx),
                 DtoGEP(st, val, 0, idx, ".vthis"));
}

} // anonymous namespace

static void DtoCreateNestedContextType(FuncDeclaration *fd);

DValue *DtoNestedVariable(const Loc &loc, Type *astype, VarDeclaration *vd,
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
  } else if (AggregateDeclaration *ad = irfunc->decl->isMember2()) {
    Logger::println(
        "Current function is member of nested class, loading vthis");
    LLValue *val = loadThisPtr(ad, *irfunc);

    for (; ad; ad = ad->toParent2()->isAggregateDeclaration()) {
      assert(ad->vthis);
      val = indexVThis(ad, val);
    }
    ctx = val;
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
  IF_LOG { Logger::cout() << "casting to: " << *irfunc->frameType << '\n'; }
  LLValue *val = ctx;
  llvm::StructType *currFrame = irfunc->frameType;
  // Make the DWARF variable address relative to the context pointer (ctx);
  // register all ops (offsetting, dereferencing) required to get there in the
  // following list.
  LLSmallVector<uint64_t, 4> dwarfAddrOps;

  const auto offsetToNthField = [&val, &dwarfAddrOps, &currFrame](unsigned fieldIndex,
                                                      const char *name = "") {
    gIR->DBuilder.OpOffset(dwarfAddrOps, currFrame, fieldIndex);
    val = DtoGEP(currFrame, val, 0, fieldIndex, name);
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
    currFrame = getIrFunc(fd)->frameType;
    gIR->DBuilder.OpDeref(dwarfAddrOps);
    val = DtoAlignedLoad(currFrame->getPointerTo(), val,
                         (std::string(".frame.") + vdparent->toChars()).c_str());
    IF_LOG Logger::cout() << "Frame: " << *val << '\n';
  }

  offsetToNthField(irLocal->nestedIndex, vd->toChars());
  IF_LOG {
    Logger::cout() << "Addr: " << *val << '\n';
    Logger::cout() << "of type: " << *val->getType() << '\n';
  }
  if (isSpecialRefVar(vd)) {
    // Handled appropriately by makeVarDValue() and EmitLocalVariable(), pass
    // storage of pointer (reference lvalue).
  } else if (byref || captureByRef(vd)) {
    val = DtoAlignedLoad(irLocal->value->getType(), val);
    // ref/out variables get a reference-debuginfo-type in EmitLocalVariable()
    // => don't dereference, use reference lvalue as address
    if (!vd->isReference())
      gIR->DBuilder.OpDeref(dwarfAddrOps);
    IF_LOG {
      Logger::cout() << "Was byref, now: " << *irLocal->value << '\n';
      Logger::cout() << "of type: " << *irLocal->value->getType() << '\n';
    }
  }

  if (!skipDIDeclaration && global.params.symdebug) {
    gIR->DBuilder.EmitLocalVariable(ctx, vd, nullptr, false,
                                    /*forceAsLocal=*/true, false, dwarfAddrOps);
  }

  return makeVarDValue(astype, vd, val);
}

void DtoResolveNestedContext(const Loc &loc, AggregateDeclaration *decl,
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
    llvm::StructType *st = getIrAggr(decl)->getLLStructType();
    LLValue *gep = DtoGEP(st, value, 0, idx, ".vthis");
    DtoStore(nest, gep);
  }
}

LLValue *DtoNestedContext(const Loc &loc, Dsymbol *sym) {
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
    val = loadThisPtr(ad, irFunc);
    if (!ad->vthis) {
      // This is just a plain 'outer' reference of a class nested in a
      // function (but without any variables in the nested context).
      return val;
    }
    val = indexVThis(ad, val);
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

      /*
       The following can happen with
       class Outer {
           class Inner {
               auto opSlice() {
                   struct Range {
                       void foo() {}
                   }
                   return Range();
               }
           }
       }
       see https://github.com/ldc-developers/ldc/issues/4130
       */
      if (!ctxfd)
        ctxfd = irFunc.decl;
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
      llvm::StructType *type = getIrFunc(ctxfd)->frameType;
      val = DtoGEP(type, val, 0, neededDepth);
      val = DtoAlignedLoad(type->getElementType(neededDepth),
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

  FuncDeclaration *parentFunc = getParentFunc(fd);
  // Make sure the parent has already been analyzed.
  if (parentFunc) {
    DtoCreateNestedContextType(parentFunc);
  }

  DtoDeclareFunction(fd);

  IrFunction &irFunc = *getIrFunc(fd);

  if (irFunc.nestedContextCreated) {
    Logger::println("already done");
    return;
  }
  irFunc.nestedContextCreated = true;

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

  AggrTypeBuilder builder;

  if (depth != 0) {
    assert(innerFrameType);
    // Add frame pointer types for all but last frame
    for (unsigned i = 0; i < (depth - 1); ++i) {
      builder.addType(innerFrameType->getElementType(i), target.ptrsize);
    }
    // Add frame pointer type for last frame
    builder.addType(LLPointerType::getUnqual(innerFrameType), target.ptrsize);
  }

  // Add the direct nested variables of this function, and update their
  // indices to match.
  // TODO: optimize ordering for minimal space usage?
  unsigned maxAlignment = 1;
  for (auto vd : fd->closureVars) {
    const bool isParam = vd->isParameter();

    LLType *t = nullptr;
    unsigned alignment = 0;
    if (captureByRef(vd)) {
      t = DtoType(pointerTo(vd->type));
      alignment = target.ptrsize;
    } else if (isParam && (vd->storage_class & STClazy)) {
      // the type is a delegate (LL struct)
      auto tf = TypeFunction::create(nullptr, vd->type, VARARGnone, LINK::d);
      auto td = TypeDelegate::create(tf);
      t = DtoType(td);
      alignment = target.ptrsize;
    } else {
      t = DtoMemType(vd->type);
      alignment = DtoAlignment(vd);
    }

    if (alignment > 1) {
      builder.alignCurrentOffset(alignment);
      maxAlignment = std::max(maxAlignment, alignment);
    }

    IrLocal &irLocal =
        *(isParam ? getIrParameter(vd, true) : getIrLocal(vd, true));
    irLocal.nestedIndex = builder.currentFieldIndex();
    irLocal.nestedDepth = depth;

    builder.addType(t, getTypeAllocSize(t));

    IF_LOG Logger::cout() << "Nested var '" << vd->toChars() << "' of type "
                          << *t << "\n";
  }

  LLStructType *frameType = LLStructType::create(
      gIR->context(), builder.defaultTypes(),
      std::string("nest.") + fd->toChars(), builder.isPacked());

  IF_LOG Logger::cout() << "frameType = " << *frameType << '\n';

  // Store type in IrFunction
  irFunc.frameType = frameType;
  irFunc.frameTypeAlignment = maxAlignment;
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
    const unsigned frameAlignment =
        std::max(getABITypeAlign(frameType), irFunc.frameTypeAlignment);

    // Create frame for current function and append to frames list
    LLValue *frame = nullptr;
    bool needsClosure = fd->needsClosure();
    IF_LOG Logger::println("Needs closure (GC) flag: %d", (int)needsClosure);
    if (needsClosure) {
      LLFunction *fn =
          getRuntimeFunction(fd->loc, gIR->module, "_d_allocmemory");
      auto size = getTypeAllocSize(frameType);
      if (frameAlignment > 16) // GC guarantees an alignment of 16
        size += frameAlignment - 16;
      LLValue *mem =
          gIR->CreateCallOrInvoke(fn, DtoConstSize_t(size), ".gc_frame");
      if (frameAlignment <= 16) {
        frame = mem;
      } else {
        const uint64_t mask = frameAlignment - 1;
        mem = gIR->ir->CreatePtrToInt(mem, DtoSize_t());
        mem = gIR->ir->CreateAdd(mem, DtoConstSize_t(mask));
        mem = gIR->ir->CreateAnd(mem, DtoConstSize_t(~mask));
        frame =
            gIR->ir->CreateIntToPtr(mem, frameType->getPointerTo(), ".frame");
      }
    } else {
      frame = DtoRawAlloca(frameType, frameAlignment, ".frame");
    }

    // copy parent frames into beginning
    if (depth != 0) {
      LLValue *src = irFunc.nestArg;
      if (!src) {
        assert(irFunc.thisArg);
        AggregateDeclaration *ad = fd->isMember2();
        assert(ad);
        assert(ad->vthis);
        LLValue *thisptr = loadThisPtr(ad, irFunc);
        IF_LOG Logger::println("Indexing to 'this'");
        src = indexVThis(ad, thisptr);
      }
      if (depth > 1) {
        DtoMemCpy(frame, src, DtoConstSize_t((depth - 1) * target.ptrsize),
                  getABITypeAlign(getVoidPtrType()));
      }
      // Copy nestArg into framelist; the outer frame is not in the list of
      // pointers
      LLValue *gep = DtoGEP(frameType, frame, 0, depth - 1);
      DtoAlignedStore(src, gep);
    }

    funcGen.nestedVar = frame;

    // go through all nested vars and assign addresses where possible.
    for (auto vd : fd->closureVars) {
      if (needsClosure && vd->needsScopeDtor()) {
        // This should really be a front-end, not a glue layer error,
        // but we need to fix this in DMD too.
        error(vd->loc, "%s `%s` has scoped destruction, cannot build closure",
              vd->kind(), vd->toPrettyChars());
      }

      IrLocal *irLocal = getIrLocal(vd);
      LLValue *gep = DtoGEP(frameType, frame, 0, irLocal->nestedIndex, vd->toChars());
      if (vd->isParameter()) {
        IF_LOG Logger::println("nested param: %s", vd->toChars());
        LOG_SCOPE
        IrParameter *parm = getIrParameter(vd);
        assert(parm->value);
        assert(parm->value->getType()->isPointerTy());

        if (vd->isReference()) {
          Logger::println(
              "Captured by reference, copying pointer to nested frame");
          DtoAlignedStore(parm->value, gep);
          // pass GEP as reference lvalue to EmitLocalVariable()
        } else {
          Logger::println("Moving to nested frame");
          // The parameter value is an alloca'd stack slot.
          // Copy to the nesting frame and leave the alloca for
          // the optimizers to clean up.
          DtoMemCpy(frameType->getContainedType(irLocal->nestedIndex), gep, parm->value);
          gep->takeName(parm->value);
          parm->value = gep; // update variable lvalue
        }
      } else if (isNRVOVar(vd)) {
        IF_LOG Logger::println(
            "nested NRVO var: %s, copying pointer to nested frame",
            vd->toChars());
        assert(irFunc.sretArg);
        DtoAlignedStore(irFunc.sretArg, gep);
        assert(!irLocal->value);
        irLocal->value = irFunc.sretArg;
        gep = irFunc.sretArg; // lvalue for debuginfo
      } else {
        IF_LOG Logger::println("nested var: %s, allocating in nested frame",
                               vd->toChars());
        assert(!irLocal->value);
        irLocal->value = gep;
      }

      gIR->DBuilder.EmitLocalVariable(gep, vd);
    }
  }
}
