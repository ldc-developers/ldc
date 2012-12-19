//===-- nested.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/nested.h"

#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/functions.h"
#include "gen/todebug.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
namespace cl = llvm::cl;

/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// NESTED VARIABLE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

static FuncDeclaration* getParentFunc(Dsymbol* sym, bool stopOnStatic) {
    if (!sym)
        return NULL;
    Dsymbol* parent = sym->parent;
    assert(parent);
    while (parent && !parent->isFuncDeclaration()) {
        if (stopOnStatic) {
            Declaration* decl = sym->isDeclaration();
            if (decl && decl->isStatic())
                return NULL;
        }
        parent = parent->parent;
    }

    return (parent ? parent->isFuncDeclaration() : NULL);
}

static void storeVariable(VarDeclaration *vd, LLValue *dst)
{
    LLValue *value = vd->ir.irLocal->value;
#if DMDV2
    int ty = vd->type->ty;
    FuncDeclaration *fd = getParentFunc(vd, true);
    assert(fd && "No parent function for nested variable?");
    if (fd->needsClosure() && !vd->isRef() && (ty == Tstruct || ty == Tsarray) && isaPointer(value->getType())) {
        // Copy structs and static arrays
        LLValue *mem = DtoGcMalloc(DtoType(vd->type), ".gc_mem");
        DtoAggrCopy(mem, value);
        DtoAlignedStore(mem, dst);
    } else
#endif
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
    LLType *int64Ty = LLType::getInt64Ty(gIR->context());

    // get the nested context
    LLValue* ctx = 0;
    if (irfunc->decl->isMember2())
    {
    #if DMDV2
        AggregateDeclaration* cd = irfunc->decl->isMember2();
        LLValue* val = irfunc->thisArg;
        if (cd->isClassDeclaration())
            val = DtoLoad(val);
        ctx = DtoLoad(DtoGEPi(val, 0, cd->vthis->ir.irField->index, ".vthis"));
    #else
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        LLValue* val = DtoLoad(irfunc->thisArg);
        ctx = DtoGEPi(val, 0, cd->vthis->ir.irField->index, ".vthis");

        if (!irfunc->frameType && vd->isThisDeclaration())
        {
            // If the only "nested" variable is the outer this pointer, we don't
            // emit a normal context, but just store the this pointer - see
            // GitHub #127.
            return new DVarValue(astype, vd, ctx);
        }

        ctx = DtoLoad(ctx);
    #endif
    }
    else if (irfunc->nestedVar) {
        ctx = irfunc->nestedVar;
        dwarfValue = ctx;
    } else {
        ctx = DtoLoad(irfunc->nestArg);
        dwarfValue = irfunc->nestArg;
        if (global.params.symdebug)
            dwarfOpDeref(dwarfAddr);
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
            dwarfOpOffset(dwarfAddr, val, vd->ir.irLocal->nestedDepth);
            dwarfOpDeref(dwarfAddr);
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
        dwarfOpOffset(dwarfAddr, val, idx);

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
        DtoDwarfLocalVariable(dwarfValue, vd, dwarfAddr);

    return new DVarValue(astype, vd, val);
}

#if DMDV2
void DtoResolveNestedContext(Loc loc, AggregateDeclaration *decl, LLValue *value)
#else
void DtoResolveNestedContext(Loc loc, ClassDeclaration *decl, LLValue *value)
#endif
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
#if DMDV2
        AggregateDeclaration* ad = irfunc->decl->isMember2();
        val = ad->isClassDeclaration() ? DtoLoad(irfunc->thisArg) : irfunc->thisArg;
        if (!ad->vthis)
            return llvm::UndefValue::get(getVoidPtrType());
#else
        ClassDeclaration* ad = irfunc->decl->isMember2()->isClassDeclaration();
        val = DtoLoad(irfunc->thisArg);
        if (!ad || !ad->vthis)
            return val;
#endif
        val = DtoLoad(DtoGEPi(val, 0, ad->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        // Use null instead of e.g. LLVM's undef to not break bitwise
        // comparison for instances of nested struct types which don't have any
        // nested references.
        return llvm::ConstantPointerNull::get(getVoidPtrType());
    }

    struct FuncDeclaration* fd = 0;
#if DMDV2
    if (AggregateDeclaration *ad = sym->isAggregateDeclaration())
        // If sym is a nested struct or a nested class, pass the frame
        // of the function where sym is declared.
        fd = ad->toParent()->isFuncDeclaration();
    else
#endif
    if (FuncDeclaration* symfd = sym->isFuncDeclaration()) {
        // Make sure we've had a chance to analyze nested context usage
#if DMDV2
        DtoCreateNestedContextType(symfd);
#else
        DtoDefineFunction(symfd);
#endif

        // if this is for a function that doesn't access variables from
        // enclosing scopes, it doesn't matter what we pass.
        // Tell LLVM about it by passing an 'undef'.
        if (symfd && symfd->ir.irFunc->depth == -1)
            return llvm::UndefValue::get(getVoidPtrType());

        // If sym is a nested function, and it's parent context is different than the
        // one we got, adjust it.
        fd = getParentFunc(symfd, true);
    }
    if (fd) {
        Logger::println("For nested function, parent is %s", fd->toChars());
        FuncDeclaration* ctxfd = irfunc->decl;
        Logger::println("Current function is %s", ctxfd->toChars());
        if (fromParent) {
            ctxfd = getParentFunc(ctxfd, true);
            assert(ctxfd && "Context from outer function, but no outer function?");
        }
        Logger::println("Context is from %s", ctxfd->toChars());

        unsigned neededDepth = fd->ir.irFunc->depth;
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
            val = DtoAlignedLoad(val, (std::string(".frame.") + fd->toChars()).c_str());
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

#if DMDV2
    if (fd->nestedVars.empty()) {
        // fill nestedVars
        size_t nnest = fd->closureVars.dim;
        for (size_t i = 0; i < nnest; ++i)
        {
            VarDeclaration* vd = static_cast<VarDeclaration*>(fd->closureVars.data[i]);
            fd->nestedVars.insert(vd);
        }
    }
#endif

    // construct nested variables array
    if (!fd->nestedVars.empty())
    {
        Logger::println("has nested frame");
        // start with adding all enclosing parent frames until a static parent is reached

        LLStructType* innerFrameType = NULL;
        unsigned depth = -1;
        if (!fd->isStatic()) {
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
        for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i)
        {
            VarDeclaration* vd = *i;
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
                        types.push_back(DtoType(vd->type));
                } else {
                    types.push_back(irparam->value->getType());
                }
            } else if (isSpecialRefVar(vd)) {
                types.push_back(DtoType(vd->type->pointerTo()));
            } else {
                types.push_back(DtoType(vd->type));
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
    if (!fd->nestedVars.empty())
    {
        IrFunction* irfunction = fd->ir.irFunc;
        unsigned depth = irfunction->depth;
        LLStructType *frameType = irfunction->frameType;
        // Create frame for current function and append to frames list
        // FIXME: alignment ?
        LLValue* frame = 0;
#if DMDV2
        if (fd->needsClosure())
            frame = DtoGcMalloc(frameType, ".frame");
        else
#endif
        frame = DtoRawAlloca(frameType, 0, ".frame");


        // copy parent frames into beginning
        if (depth != 0) {
            LLValue* src = irfunction->nestArg;
            if (!src) {
                assert(irfunction->thisArg);
                assert(fd->isMember2());
                LLValue* thisval = DtoLoad(irfunction->thisArg);
#if DMDV2
                AggregateDeclaration* cd = fd->isMember2();
#else
                ClassDeclaration* cd = fd->isMember2()->isClassDeclaration();
#endif
                assert(cd);
                assert(cd->vthis);
                Logger::println("Indexing to 'this'");
#if DMDV2
                if (cd->isStructDeclaration())
                    src = DtoExtractValue(thisval, cd->vthis->ir.irField->index, ".vthis");
                else
#endif
                src = DtoLoad(DtoGEPi(thisval, 0, cd->vthis->ir.irField->index, ".vthis"));
            } else {
                src = DtoLoad(src);
            }
            if (depth > 1) {
                src = DtoBitCast(src, getVoidPtrType());
                LLValue* dst = DtoBitCast(frame, getVoidPtrType());
                DtoMemCpy(dst, src, DtoConstSize_t((depth-1) * PTRSIZE),
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
        for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i)
        {
            VarDeclaration* vd = *i;

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
                dwarfOpOffset(addr, frameType, vd->ir.irLocal->nestedIndex);
                DtoDwarfLocalVariable(frame, vd, addr);
            }
        }
    } else if (FuncDeclaration* parFunc = getParentFunc(fd, true)) {
        // Propagate context arg properties if the context arg is passed on unmodified.
        DtoDeclareFunction(parFunc);
        fd->ir.irFunc->frameType = parFunc->ir.irFunc->frameType;
        fd->ir.irFunc->depth = parFunc->ir.irFunc->depth;
    }
}
