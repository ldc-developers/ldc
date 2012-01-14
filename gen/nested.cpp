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

/// What the context pointer for a nested function looks like
enum NestedCtxType {
    /// Context is void*[] of pointers to variables.
    /// Variables from higher levels are at the front.
    NCArray,

    /// Context is a struct containing variables belonging to the parent function.
    /// If the parent function itself has a parent function, one of the members is
    /// a pointer to its context. (linked-list style)
    // FIXME: implement
    // TODO: Functions without any variables accessed by nested functions, but
    //       with a parent whose variables are accessed, can use the parent's
    //       context.
    // NOTE: This is what DMD seems to do.
    NCStruct,

    /// Context is a list of pointers to structs of variables, followed by the
    /// variables of the inner-most function with variables accessed by nested
    /// functions. The initial pointers point to similar structs for enclosing
    /// functions.
    /// Only functions whose variables are accessed by nested functions create
    /// new frames, others just pass on what got passed in.
    NCHybrid
};

static cl::opt<NestedCtxType> nestedCtx("nested-ctx",
    cl::desc("How to construct a nested function's context:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(NCArray,  "array",  "Array of pointers to variables (including multi-level)"),
        //clEnumValN(NCStruct, "struct", "Struct of variables (with multi-level via linked list)"),
        clEnumValN(NCHybrid, "hybrid", "List of pointers to structs of variables, one per level."),
        clEnumValEnd),
    cl::init(NCHybrid));


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
    #else
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        LLValue* val = DtoLoad(irfunc->thisArg);
    #endif
        ctx = DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
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

    if (nestedCtx == NCArray) {
        LLValue* val = DtoBitCast(ctx, getPtrToType(getVoidPtrType()));
        val = DtoGEPi1(val, vd->ir.irLocal->nestedIndex);
        val = DtoAlignedLoad(val);
        assert(vd->ir.irLocal->value);
        val = DtoBitCast(val, vd->ir.irLocal->value->getType(), vd->toChars());
        return new DVarValue(astype, vd, val);
    }
    else if (nestedCtx == NCHybrid) {
        LLValue* val = DtoBitCast(ctx, LLPointerType::getUnqual(irfunc->frameType));
        Logger::cout() << "Context: " << *val << '\n';
        Logger::cout() << "of type: " << *val->getType() << '\n';

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

        if (dwarfValue && global.params.symdebug)
            dwarfOpOffset(dwarfAddr, val, vd->ir.irLocal->nestedIndex);
        val = DtoGEPi(val, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
        Logger::cout() << "Addr: " << *val << '\n';
        Logger::cout() << "of type: " << *val->getType() << '\n';
        if (vd->ir.irLocal->byref || byref) {
            val = DtoAlignedLoad(val);
            //dwarfOpDeref(dwarfAddr);
            Logger::cout() << "Was byref, now: " << *val << '\n';
            Logger::cout() << "of type: " << *val->getType() << '\n';
        }

        if (dwarfValue && global.params.symdebug)
            DtoDwarfLocalVariable(dwarfValue, vd, dwarfAddr);

        return new DVarValue(astype, vd, val);
    }
    else {
        assert(0 && "Not implemented yet");
    }
}

void DtoNestedInit(VarDeclaration* vd)
{
    Logger::println("DtoNestedInit for %s", vd->toChars());
    LOG_SCOPE

    IrFunction* irfunc = gIR->func()->decl->ir.irFunc;
    LLValue* nestedVar = irfunc->nestedVar;

    if (nestedCtx == NCArray) {
        // alloca as usual if no value already
        if (!vd->ir.irLocal->value)
            vd->ir.irLocal->value = DtoAlloca(vd->type, vd->toChars());

        // store the address into the nested vars array
        assert(vd->ir.irLocal->nestedIndex >= 0);
        LLValue* gep = DtoGEPi(nestedVar, 0, vd->ir.irLocal->nestedIndex);

        assert(isaPointer(vd->ir.irLocal->value));
        LLValue* val = DtoBitCast(vd->ir.irLocal->value, getVoidPtrType());

        DtoAlignedStore(val, gep);
    }
    else if (nestedCtx == NCHybrid) {
        assert(vd->ir.irLocal->value && "Nested variable without storage?");

        if (!vd->isParameter() && (vd->isRef() || vd->isOut())) {
            unsigned vardepth = vd->ir.irLocal->nestedDepth;

            LLValue* val = NULL;
            // Retrieve frame pointer
            if (vardepth == irfunc->depth) {
                val = nestedVar;
            } else {
                FuncDeclaration *parentfunc = getParentFunc(vd, true);
                assert(parentfunc && "No parent function for nested variable?");

                val = DtoGEPi(nestedVar, 0, vardepth);
                val = DtoAlignedLoad(val, (std::string(".frame.") + parentfunc->toChars()).c_str());
            }
            val = DtoGEPi(val, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
            storeVariable(vd, val);
        } else {
            // Already initialized in DtoCreateNestedContext
        }
    }
    else {
        assert(0 && "Not implemented yet");
    }
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
#else
        ClassDeclaration* ad = irfunc->decl->isMember2()->isClassDeclaration();
        val = DtoLoad(irfunc->thisArg);
#endif
        if (!ad || !ad->vthis)
            return llvm::UndefValue::get(getVoidPtrType());
        val = DtoLoad(DtoGEPi(val, 0,ad->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        return llvm::UndefValue::get(getVoidPtrType());
    }
    if (nestedCtx == NCHybrid) {
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
    }
    Logger::cout() << "result = " << *val << '\n';
    Logger::cout() << "of type " << *val->getType() << '\n';
    return val;
}

static void DtoCreateNestedContextType(FuncDeclaration* fd) {
    Logger::println("DtoCreateNestedContextType for %s", fd->toChars());
    LOG_SCOPE

#if DMDV2
    DtoDeclareFunction(fd);
#endif

    if (fd->ir.irFunc->nestedContextCreated)
        return;
    fd->ir.irFunc->nestedContextCreated = true;

#if DMDV2
    if (fd->nestedVars.empty()) {
        // fill nestedVars
        size_t nnest = fd->closureVars.dim;
        for (size_t i = 0; i < nnest; ++i)
        {
            VarDeclaration* vd = (VarDeclaration*)fd->closureVars.data[i];
            fd->nestedVars.insert(vd);
        }
    }
#endif

    if (nestedCtx == NCHybrid) {
        // construct nested variables array
        if (!fd->nestedVars.empty())
        {
            Logger::println("has nested frame");
            // start with adding all enclosing parent frames until a static parent is reached

            LLStructType* innerFrameType = NULL;
            unsigned depth = -1;
            if (!fd->isStatic()) {
                if (FuncDeclaration* parfd = getParentFunc(fd, true)) {
                    // Make sure parfd->ir.irFunc has already been set.
                    DtoDeclareFunction(parfd);

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

            if (Logger::enabled()) {
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
                    assert(vd->ir.irParam->value);
                    LLValue* value = vd->ir.irParam->value;
                    LLType* type = value->getType();
                    bool refout = vd->storage_class & (STCref | STCout);
                    bool lazy = vd->storage_class & STClazy;
                    if (!refout && (!vd->ir.irParam->arg->byref || lazy)) {
                        // This will be copied to the nesting frame.
                        if (lazy)
                            type = type->getContainedType(0);
                        else
                            type = DtoType(vd->type);
                        vd->ir.irParam->byref = false;
                    } else {
                        vd->ir.irParam->byref = true;
                    }
                    types.push_back(type);
                } else if (vd->isRef() || vd->isOut()) {
                    // Foreach variables can also be by reference, for instance.
                    types.push_back(DtoType(vd->type->pointerTo()));
                    vd->ir.irLocal->byref = true;
                } else {
                    types.push_back(DtoType(vd->type));
                    vd->ir.irLocal->byref = false;
                }
                if (Logger::enabled()) {
                    Logger::println("Nested var: %s", vd->toChars());
                    Logger::cout() << "of type: " << *types.back() << '\n';
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
    else {
        assert(0 && "Not implemented yet");
    }
}


void DtoCreateNestedContext(FuncDeclaration* fd) {
    Logger::println("DtoCreateNestedContext for %s", fd->toChars());
    LOG_SCOPE

    DtoCreateNestedContextType(fd);

    if (nestedCtx == NCArray) {
        // construct nested variables array
        if (!fd->nestedVars.empty())
        {
            Logger::println("has nested frame");
            // start with adding all enclosing parent frames until a static parent is reached
            int nparelems = 0;
            if (!fd->isStatic())
            {
                Dsymbol* par = fd->toParent2();
                while (par)
                {
                    if (FuncDeclaration* parfd = par->isFuncDeclaration())
                    {
                        nparelems += parfd->nestedVars.size();
                        // stop at first static
                        if (parfd->isStatic())
                            break;
                    }
                    else if (par->isClassDeclaration())
                    {
                        // nothing needed
                    }
                    else
                    {
                        break;
                    }

                    par = par->toParent2();
                }
            }
            int nelems = fd->nestedVars.size() + nparelems;

            // make array type for nested vars
            LLType* nestedVarsTy = LLArrayType::get(getVoidPtrType(), nelems);

            // alloca it
            // FIXME align ?
            LLValue* nestedVars = DtoRawAlloca(nestedVarsTy, 0, ".nested_vars");

            IrFunction* irfunction = fd->ir.irFunc;

            // copy parent frame into beginning
            if (nparelems)
            {
                LLValue* src = irfunction->nestArg;
                if (!src)
                {
                    assert(irfunction->thisArg);
                    assert(fd->isMember2());
                    LLValue* thisval = DtoLoad(irfunction->thisArg);
                    ClassDeclaration* cd = fd->isMember2()->isClassDeclaration();
                    assert(cd);
                    assert(cd->vthis);
                    src = DtoLoad(DtoGEPi(thisval, 0,cd->vthis->ir.irField->index, ".vthis"));
                } else {
                    src = DtoLoad(src);
                }
                DtoMemCpy(nestedVars, src, DtoConstSize_t(nparelems*PTRSIZE),
                    getABITypeAlign(getVoidPtrType()));
            }

            // store in IrFunction
            irfunction->nestedVar = nestedVars;

            // go through all nested vars and assign indices
            int idx = nparelems;
            for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i)
            {
                VarDeclaration* vd = *i;
                if (!vd->ir.irLocal)
                    vd->ir.irLocal = new IrLocal(vd);

                if (vd->isParameter())
                {
                    Logger::println("nested param: %s", vd->toChars());
                    LLValue* gep = DtoGEPi(nestedVars, 0, idx);
                    LLValue* val = DtoBitCast(vd->ir.irLocal->value, getVoidPtrType());
                    DtoAlignedStore(val, gep);
                }
                else
                {
                    Logger::println("nested var:   %s", vd->toChars());
                }

                vd->ir.irLocal->nestedIndex = idx++;
            }
        }
    }
    else if (nestedCtx == NCHybrid) {
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
                    LLValue* value = vd->ir.irLocal->value;
                    if (llvm::isa<llvm::AllocaInst>(llvm::GetUnderlyingObject(value))) {
                        Logger::println("Copying to nested frame");
                        // The parameter value is an alloca'd stack slot.
                        // Copy to the nesting frame and leave the alloca for
                        // the optimizers to clean up.
                        assert(!vd->ir.irLocal->byref);
                        DtoStore(DtoLoad(value), gep);
                        gep->takeName(value);
                        vd->ir.irLocal->value = gep;
                    } else {
                        Logger::println("Adding pointer to nested frame");
                        // The parameter value is something else, such as a
                        // passed-in pointer (for 'ref' or 'out' parameters) or
                        // a pointer arg with byval attribute.
                        // Store the address into the frame.
                        assert(vd->ir.irLocal->byref);
                        storeVariable(vd, gep);
                    }
                } else if (vd->isRef() || vd->isOut()) {
                    // This slot is initialized in DtoNestedInit, to handle things like byref foreach variables
                    // which move around in memory.
                    assert(vd->ir.irLocal->byref);
                } else {
                    Logger::println("nested var:   %s", vd->toChars());
                    if (vd->ir.irLocal->value)
                        Logger::cout() << "Pre-existing value: " << *vd->ir.irLocal->value << '\n';
                    assert(!vd->ir.irLocal->value);
                    vd->ir.irLocal->value = gep;
                    assert(!vd->ir.irLocal->byref);
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
    else {
        assert(0 && "Not implemented yet");
    }
}
