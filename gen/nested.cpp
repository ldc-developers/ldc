//===-- nested.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "target.h"
#include "gen/nested.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
namespace cl = llvm::cl;

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// NESTED VARIABLE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

static void storeVariable(VarDeclaration *vd, LLValue *dst)
{
    LLValue *value = vd->ir.irLocal->value;
    int ty = vd->type->ty;
    FuncDeclaration *fd = getParentFunc(vd, true);
    assert(fd && "No parent function for nested variable?");
    if (fd->needsClosure() && !vd->isRef() && (ty == Tstruct || ty == Tsarray) && isaPointer(value->getType())) {
        // Copy structs and static arrays
        LLValue *mem = DtoGcMalloc(DtoType(vd->type), ".gc_mem");
        DtoAggrCopy(mem, value);
        DtoAlignedStore(mem, dst);
    } else
    // Store the address into the frame
    DtoAlignedStore(value, dst);
}

static void DtoCreateNestedContextType(FuncDeclaration* fd);

DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd, bool byref)
{
    Logger::println("DtoNestedVariable for %s @ %s", vd->toChars(), loc.toChars());
    LOG_SCOPE;

    ////////////////////////////////////
    // Locate context value

    Dsymbol* vdparent = vd->toParent2();
    assert(vdparent);

    IrFunction* irfunc = gIR->func();

    // Check whether we can access the needed frame
    FuncDeclaration *fd = irfunc->decl;
    while (fd != vdparent) {
        if (fd->isStatic()) {
            error(loc, "function %s cannot access frame of function %s", irfunc->decl->toPrettyChars(), vdparent->toPrettyChars());
            return new DVarValue(astype, vd, llvm::UndefValue::get(getPtrToType(DtoType(astype))));
        }
        fd = getParentFunc(fd, false);
        assert(fd);
    }

    // is the nested variable in this scope?
    if (vdparent == irfunc->decl)
    {
        LLValue* val = vd->ir.getIrValue();
        return new DVarValue(astype, vd, val);
    }

    LLValue *dwarfValue = 0;
    std::vector<LLValue*> dwarfAddr;

    // get the nested context
    LLValue* ctx = 0;
    if (irfunc->nestedVar) {
        // If this function has its own nested context struct, always load it.
        ctx = irfunc->nestedVar;
        dwarfValue = ctx;
    } else if (irfunc->decl->isMember2()) {
        // If this is a member function of a nested class without its own
        // context, load the vthis member.
        AggregateDeclaration* cd = irfunc->decl->isMember2();
        LLValue* val = irfunc->thisArg;
        if (cd->isClassDeclaration())
            val = DtoLoad(val);
        ctx = DtoLoad(DtoGEPi(val, 0, cd->vthis->ir.irField->index, ".vthis"));
    } else {
        // Otherwise, this is a simple nested function, load from the context
        // argument.
        ctx = DtoLoad(irfunc->nestArg);
        dwarfValue = irfunc->nestArg;
        if (global.params.symdebug)
            gIR->DBuilder.OpDeref(dwarfAddr);
    }
    assert(ctx);

    DtoCreateNestedContextType(vdparent->isFuncDeclaration());
    assert(vd->ir.irLocal);

    ////////////////////////////////////
    // Extract variable from nested context

    LLValue* val = DtoBitCast(ctx, LLPointerType::getUnqual(irfunc->frameType));
    Logger::cout() << "Context: " << *val << '\n';
    Logger::cout() << "of type: " << *irfunc->frameType << '\n';

    unsigned vardepth = vd->ir.irLocal->nestedDepth;
    unsigned funcdepth = irfunc->depth;

    Logger::cout() << "Variable: " << vd->toChars() << '\n';
    Logger::cout() << "Variable depth: " << vardepth << '\n';
    Logger::cout() << "Function: " << irfunc->decl->toChars() << '\n';
    Logger::cout() << "Function depth: " << funcdepth << '\n';

    if (vardepth == funcdepth) {
        // This is not always handled above because functions without
        // variables accessed by nested functions don't create new frames.
        Logger::println("Same depth");
    } else {
        // Load frame pointer and index that...
        if (dwarfValue && global.params.symdebug) {
            gIR->DBuilder.OpOffset(dwarfAddr, val, vd->ir.irLocal->nestedDepth);
            gIR->DBuilder.OpDeref(dwarfAddr);
        }
        Logger::println("Lower depth");
        val = DtoGEPi(val, 0, vd->ir.irLocal->nestedDepth);
        Logger::cout() << "Frame index: " << *val << '\n';
        val = DtoAlignedLoad(val, (std::string(".frame.") + vdparent->toChars()).c_str());
        Logger::cout() << "Frame: " << *val << '\n';
    }

    int idx = vd->ir.irLocal->nestedIndex;
    assert(idx != -1 && "Nested context not yet resolved for variable.");

    if (dwarfValue && global.params.symdebug)
        gIR->DBuilder.OpOffset(dwarfAddr, val, idx);

    val = DtoGEPi(val, 0, idx, vd->toChars());
    Logger::cout() << "Addr: " << *val << '\n';
    Logger::cout() << "of type: " << *val->getType() << '\n';
    if (byref || (vd->isParameter() && vd->ir.irParam->arg->byref)) {
        val = DtoAlignedLoad(val);
        //dwarfOpDeref(dwarfAddr);
        Logger::cout() << "Was byref, now: " << *val << '\n';
        Logger::cout() << "of type: " << *val->getType() << '\n';
    }

    if (dwarfValue && global.params.symdebug)
        gIR->DBuilder.EmitLocalVariable(dwarfValue, vd, dwarfAddr);

    return new DVarValue(astype, vd, val);
}

void DtoResolveNestedContext(Loc loc, AggregateDeclaration *decl, LLValue *value)
{
    Logger::println("Resolving nested context");
    LOG_SCOPE;

    // get context
    LLValue* nest = DtoNestedContext(loc, decl);

    // store into right location
    if (!llvm::dyn_cast<llvm::UndefValue>(nest)) {
        // Need to make sure the declaration has already been resolved, because
        // when multiple source files are specified on the command line, the
        // frontend sometimes adds "nested" (i.e. a template in module B
        // instantiated from module A with a type from module A instantiates
        // another template from module B) into the wrong module, messing up
        // our codegen order.
        DtoResolveDsymbol(decl);

        size_t idx = decl->vthis->ir.irField->index;
        LLValue* gep = DtoGEPi(value,0,idx,".vthis");
        DtoStore(DtoBitCast(nest, gep->getType()->getContainedType(0)), gep);
    }
}

LLValue* DtoNestedContext(Loc loc, Dsymbol* sym)
{
    Logger::println("DtoNestedContext for %s", sym->toPrettyChars());
    LOG_SCOPE;

    // The function we are currently in, and the constructed object/called
    // function might inherit a context pointer from.
    IrFunction* irfunc = gIR->func();

    bool fromParent = true;

    LLValue* val;
    // if this func has its own vars that are accessed by nested funcs
    // use its own context
    if (irfunc->nestedVar) {
        val = irfunc->nestedVar;
        fromParent = false;
    }
    // otherwise, it may have gotten a context from the caller
    else if (irfunc->nestArg)
        val = DtoLoad(irfunc->nestArg);
    // or just have a this argument
    else if (irfunc->thisArg)
    {
        AggregateDeclaration* ad = irfunc->decl->isMember2();
        val = ad->isClassDeclaration() ? DtoLoad(irfunc->thisArg) : irfunc->thisArg;
        if (!ad->vthis)
        {
            // This is just a plain 'outer' reference of a class nested in a
            // function (but without any variables in the nested context).
            return val;
        }
        val = DtoLoad(DtoGEPi(val, 0, ad->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        // Use null instead of e.g. LLVM's undef to not break bitwise
        // comparison for instances of nested struct types which don't have any
        // nested references.
        return llvm::ConstantPointerNull::get(getVoidPtrType());
    }

    FuncDeclaration* frameToPass = 0;
    if (AggregateDeclaration *ad = sym->isAggregateDeclaration()) {
        // If sym is a nested struct or a nested class, pass the frame
        // of the function where sym is declared.
        frameToPass = ad->toParent()->isFuncDeclaration();
    } else if (FuncDeclaration* symfd = sym->isFuncDeclaration()) {
        // Make sure we've had a chance to analyze nested context usage
        DtoCreateNestedContextType(symfd);

        // if this is for a function that doesn't access variables from
        // enclosing scopes, it doesn't matter what we pass.
        if (symfd->ir.irFunc->depth == -1)
            return llvm::UndefValue::get(getVoidPtrType());

        // If sym is a nested function, and its parent context is different
        // than the one we got, adjust it.
        frameToPass = getParentFunc(symfd, true);
    }

    if (frameToPass) {
        Logger::println("Parent frame is from %s", frameToPass->toChars());
        FuncDeclaration* ctxfd = irfunc->decl;
        Logger::println("Current function is %s", ctxfd->toChars());
        if (fromParent) {
            ctxfd = getParentFunc(ctxfd, true);
            assert(ctxfd && "Context from outer function, but no outer function?");
        }
        Logger::println("Context is from %s", ctxfd->toChars());

        unsigned neededDepth = frameToPass->ir.irFunc->depth;
        unsigned ctxDepth = ctxfd->ir.irFunc->depth;

        Logger::cout() << "Needed depth: " << neededDepth << '\n';
        Logger::cout() << "Context depth: " << ctxDepth << '\n';

        if (neededDepth >= ctxDepth) {
            // assert(neededDepth <= ctxDepth + 1 && "How are we going more than one nesting level up?");
            // fd needs the same context as we do, so all is well
            Logger::println("Calling sibling function or directly nested function");
        } else {
            val = DtoBitCast(val, LLPointerType::getUnqual(ctxfd->ir.irFunc->frameType));
            val = DtoGEPi(val, 0, neededDepth);
            val = DtoAlignedLoad(val,
                (std::string(".frame.") + frameToPass->toChars()).c_str());
        }
    }

    Logger::cout() << "result = " << *val << '\n';
    Logger::cout() << "of type " << *val->getType() << '\n';
    return val;
}

static void DtoCreateNestedContextType(FuncDeclaration* fd) {
    Logger::println("DtoCreateNestedContextType for %s", fd->toChars());
    LOG_SCOPE

    DtoDeclareFunction(fd);

    if (fd->ir.irFunc->nestedContextCreated)
        return;
    fd->ir.irFunc->nestedContextCreated = true;

    // construct nested variables array
    if (fd->closureVars.dim != 0)
    {
        Logger::println("has nested frame");
        // start with adding all enclosing parent frames until a static parent is reached

        LLStructType* innerFrameType = NULL;
        unsigned depth = -1;

        // Static functions and function (not delegate) literals don't allow
        // access to a parent context, even if they are nested.
        const bool certainlyNewRoot = fd->isStatic() ||
            (fd->isFuncLiteralDeclaration() &&
            static_cast<FuncLiteralDeclaration*>(fd)->tok == TOKfunction);
        if (!certainlyNewRoot) {
            if (FuncDeclaration* parfd = getParentFunc(fd, true)) {
                // Make sure the parent has already been analyzed.
                DtoCreateNestedContextType(parfd);

                innerFrameType = parfd->ir.irFunc->frameType;
                if (innerFrameType)
                    depth = parfd->ir.irFunc->depth;
            }
        }
        fd->ir.irFunc->depth = ++depth;

        Logger::cout() << "Function " << fd->toChars() << " has depth " << depth << '\n';

        typedef std::vector<LLType*> TypeVec;
        TypeVec types;
        if (depth != 0) {
            assert(innerFrameType);
            // Add frame pointer types for all but last frame
            if (depth > 1) {
                for (unsigned i = 0; i < (depth - 1); ++i) {
                    types.push_back(innerFrameType->getElementType(i));
                }
            }
            // Add frame pointer type for last frame
            types.push_back(LLPointerType::getUnqual(innerFrameType));
        }

        if (Logger::enabled() && depth != 0) {
            Logger::println("Frame types: ");
            LOG_SCOPE;
            for (TypeVec::iterator i = types.begin(); i != types.end(); ++i)
                Logger::cout() << **i << '\n';
        }

        // Add the direct nested variables of this function, and update their indices to match.
        // TODO: optimize ordering for minimal space usage?
        VarDeclarationIter closureVarsIter(fd->closureVars);
        for (; closureVarsIter.more(); closureVarsIter.next()) {
            VarDeclaration* vd = *closureVarsIter;
            if (!vd->ir.irLocal)
                vd->ir.irLocal = new IrLocal(vd);

            vd->ir.irLocal->nestedIndex = types.size();
            vd->ir.irLocal->nestedDepth = depth;
            if (vd->isParameter()) {
                // Parameters will have storage associated with them (to handle byref etc.),
                // so handle those cases specially by storing a pointer instead of a value.
                const IrParameter* irparam = vd->ir.irParam;
                const bool refout = vd->storage_class & (STCref | STCout);
                const bool lazy = vd->storage_class & STClazy;
                const bool byref = irparam->arg->byref;
                const bool isVthisPtr = irparam->isVthis && !byref;
                if (!(refout || (byref && !lazy)) || isVthisPtr) {
                    // This will be copied to the nesting frame.
                    if (lazy)
                        types.push_back(irparam->value->getType()->getContainedType(0));
                    else
                        types.push_back(i1ToI8(DtoType(vd->type)));
                } else {
                    types.push_back(irparam->value->getType());
                }
            } else if (isSpecialRefVar(vd)) {
                types.push_back(DtoType(vd->type->pointerTo()));
            } else {
                types.push_back(i1ToI8(DtoType(vd->type)));
            }
            if (Logger::enabled()) {
                Logger::cout() << "Nested var '" << vd->toChars() <<
                    "' of type " << *types.back() << "\n";
            }
        }

        LLStructType* frameType = LLStructType::create(gIR->context(), types,
                                                       std::string("nest.") + fd->toChars());

        Logger::cout() << "frameType = " << *frameType << '\n';

        // Store type in IrFunction
        fd->ir.irFunc->frameType = frameType;
    } else if (FuncDeclaration* parFunc = getParentFunc(fd, true)) {
        // Propagate context arg properties if the context arg is passed on unmodified.
        DtoCreateNestedContextType(parFunc);
        fd->ir.irFunc->frameType = parFunc->ir.irFunc->frameType;
        fd->ir.irFunc->depth = parFunc->ir.irFunc->depth;
    }
}


void DtoCreateNestedContext(FuncDeclaration* fd) {
    Logger::println("DtoCreateNestedContext for %s", fd->toChars());
    LOG_SCOPE

    DtoCreateNestedContextType(fd);

    // construct nested variables array
    if (fd->closureVars.dim != 0)
    {
        IrFunction* irfunction = fd->ir.irFunc;
        unsigned depth = irfunction->depth;
        LLStructType *frameType = irfunction->frameType;
        // Create frame for current function and append to frames list
        // FIXME: alignment ?
        LLValue* frame = 0;
        if (fd->needsClosure())
            frame = DtoGcMalloc(frameType, ".frame");
        else
            frame = DtoRawAlloca(frameType, 0, ".frame");

        // copy parent frames into beginning
        if (depth != 0) {
            LLValue* src = irfunction->nestArg;
            if (!src) {
                assert(irfunction->thisArg);
                assert(fd->isMember2());
                LLValue* thisval = DtoLoad(irfunction->thisArg);
                AggregateDeclaration* cd = fd->isMember2();
                assert(cd);
                assert(cd->vthis);
                Logger::println("Indexing to 'this'");
                if (cd->isStructDeclaration())
                    src = DtoExtractValue(thisval, cd->vthis->ir.irField->index, ".vthis");
                else
                    src = DtoLoad(DtoGEPi(thisval, 0, cd->vthis->ir.irField->index, ".vthis"));
            } else {
                src = DtoLoad(src);
            }
            if (depth > 1) {
                src = DtoBitCast(src, getVoidPtrType());
                LLValue* dst = DtoBitCast(frame, getVoidPtrType());
                DtoMemCpy(dst, src, DtoConstSize_t((depth-1) * Target::ptrsize),
                    getABITypeAlign(getVoidPtrType()));
            }
            // Copy nestArg into framelist; the outer frame is not in the list of pointers
            src = DtoBitCast(src, frameType->getContainedType(depth-1));
            LLValue* gep = DtoGEPi(frame, 0, depth-1);
            DtoAlignedStore(src, gep);
        }

        // store context in IrFunction
        irfunction->nestedVar = frame;

        // go through all nested vars and assign addresses where possible.
        VarDeclarationIter closureVarsIter(fd->closureVars);
        for (; closureVarsIter.more(); closureVarsIter.next()) {
            VarDeclaration* vd = *closureVarsIter;

            LLValue* gep = DtoGEPi(frame, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
            if (vd->isParameter()) {
                Logger::println("nested param: %s", vd->toChars());
                LOG_SCOPE
                IrParameter* parm = vd->ir.irParam;

                if (parm->arg->byref)
                {
                    storeVariable(vd, gep);
                }
                else
                {
                    Logger::println("Copying to nested frame");
                    // The parameter value is an alloca'd stack slot.
                    // Copy to the nesting frame and leave the alloca for
                    // the optimizers to clean up.
                    DtoStore(DtoLoad(parm->value), gep);
                    gep->takeName(parm->value);
                    parm->value = gep;
                }
            } else {
                Logger::println("nested var:   %s", vd->toChars());
                assert(!vd->ir.irLocal->value);
                vd->ir.irLocal->value = gep;
            }

            if (global.params.symdebug) {
                LLSmallVector<LLValue*, 2> addr;
                gIR->DBuilder.OpOffset(addr, frameType, vd->ir.irLocal->nestedIndex);
                gIR->DBuilder.EmitLocalVariable(frame, vd, addr);
            }
        }
    }
}
