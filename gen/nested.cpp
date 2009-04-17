#include "gen/nested.h"

#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

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

DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd)
{
    Logger::println("DtoNestedVariable for %s @ %s", vd->toChars(), loc.toChars());
    LOG_SCOPE;
    
    ////////////////////////////////////
    // Locate context value
    
    Dsymbol* vdparent = vd->toParent2();
    assert(vdparent);
    
    IrFunction* irfunc = gIR->func();
    
    // is the nested variable in this scope?
    if (vdparent == irfunc->decl)
    {
        LLValue* val = vd->ir.getIrValue();
        return new DVarValue(astype, vd, val);
    }
    
    // get the nested context
    LLValue* ctx = 0;
    if (irfunc->decl->isMember2())
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        LLValue* val = DtoLoad(irfunc->thisArg);
        ctx = DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
    }
    else if (irfunc->nestedVar)
        ctx = irfunc->nestedVar;
    else
        ctx = irfunc->nestArg;
    assert(ctx);
    
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
            Logger::println("Lower depth");
            val = DtoGEPi(val, 0, vd->ir.irLocal->nestedDepth);
            Logger::cout() << "Frame index: " << *val << '\n';
            val = DtoAlignedLoad(val, (std::string(".frame.") + vdparent->toChars()).c_str());
            Logger::cout() << "Frame: " << *val << '\n';
        }
        val = DtoGEPi(val, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
        Logger::cout() << "Addr: " << *val << '\n';
        Logger::cout() << "of type: " << *val->getType() << '\n';
        if (vd->ir.irLocal->byref) {
            val = DtoAlignedLoad(val);
            Logger::cout() << "Was byref, now: " << *val << '\n';
            Logger::cout() << "of type: " << *val->getType() << '\n';
        }
        
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
            vd->ir.irLocal->value = DtoAlloca(DtoType(vd->type), vd->toChars());
        
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
                
                val = DtoGEPi(val, 0, vardepth);
                val = DtoAlignedLoad(val, (std::string(".frame.") + parentfunc->toChars()).c_str());
            }
            val = DtoGEPi(val, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
            DtoAlignedStore(vd->ir.irLocal->value, val);
        } else {
            // Already initialized in DtoCreateNestedContext
        }
    }
    else {
        assert(0 && "Not implemented yet");
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
        val = irfunc->nestArg;
    // or just have a this argument
    else if (irfunc->thisArg)
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        if (!cd || !cd->vthis)
            return llvm::UndefValue::get(getVoidPtrType());
        val = DtoLoad(irfunc->thisArg);
        val = DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        return llvm::UndefValue::get(getVoidPtrType());
    }
    if (nestedCtx == NCHybrid) {
        // If sym is a nested function, and it's parent context is different than the
        // one we got, adjust it.
        
        if (FuncDeclaration* fd = getParentFunc(sym->isFuncDeclaration(), true)) {
            // if this is for a function that doesn't access variables from
            // enclosing scopes, it doesn't matter what we pass.
            // Tell LLVM about it by passing an 'undef'.
            if (fd->ir.irFunc->depth == 0)
                return llvm::UndefValue::get(getVoidPtrType());
            
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
                assert(neededDepth <= ctxDepth + 1 && "How are we going more than one nesting level up?");
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

void DtoCreateNestedContext(FuncDeclaration* fd) {
    Logger::println("DtoCreateNestedContext for %s", fd->toChars());
    LOG_SCOPE
    
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
                    else if (ClassDeclaration* parcd = par->isClassDeclaration())
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
            const LLType* nestedVarsTy = LLArrayType::get(getVoidPtrType(), nelems);
        
            // alloca it
            LLValue* nestedVars = DtoAlloca(nestedVarsTy, ".nested_vars");
            
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
            Logger::println("has nested frame");
            // start with adding all enclosing parent frames until a static parent is reached
            
            const LLStructType* innerFrameType = NULL;
            unsigned depth = 0;
            if (!fd->isStatic()) {
                if (FuncDeclaration* parfd = getParentFunc(fd, true)) {
                    innerFrameType = parfd->ir.irFunc->frameType;
                    if (innerFrameType)
                        depth = parfd->ir.irFunc->depth + 1;
                }
            }
            fd->ir.irFunc->depth = depth;
            
            Logger::cout() << "Function " << fd->toChars() << " has depth " << depth << '\n';
            
            typedef std::vector<const LLType*> TypeVec;
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
                    // Parameters already have storage associated with them (to handle byref etc.),
                    // so handle specially for now by storing a pointer instead of a value.
                    assert(vd->ir.irLocal->value);
                    // FIXME: don't do this for normal parameters?
                    types.push_back(vd->ir.irLocal->value->getType());
                } else if (vd->isRef() || vd->isOut()) {
                    // Foreach variables can also be by reference, for instance.
                    types.push_back(DtoType(vd->type->pointerTo()));
                } else {
                    types.push_back(DtoType(vd->type));
                }
                if (Logger::enabled()) {
                    Logger::println("Nested var: %s", vd->toChars());
                    Logger::cout() << "of type: " << *types.back() << '\n';
                }
            }
            
            const LLStructType* frameType = LLStructType::get(types);
            gIR->module->addTypeName(std::string("nest.") + fd->toChars(), frameType);
            
            Logger::cout() << "frameType = " << *frameType << '\n';
            
            // Store type in IrFunction
            IrFunction* irfunction = fd->ir.irFunc;
            irfunction->frameType = frameType;
            
            // Create frame for current function and append to frames list
            // FIXME: For D2, this should be a gc_malloc (or similar) call, not alloca
            LLValue* frame = DtoAlloca(frameType, ".frame");
            
            // copy parent frames into beginning
            if (depth != 0) {
                LLValue* src = irfunction->nestArg;
                if (!src) {
                    assert(irfunction->thisArg);
                    assert(fd->isMember2());
                    LLValue* thisval = DtoLoad(irfunction->thisArg);
                    ClassDeclaration* cd = fd->isMember2()->isClassDeclaration();
                    assert(cd);
                    assert(cd->vthis);
                    Logger::println("Indexing to 'this'");
                    src = DtoLoad(DtoGEPi(thisval, 0, cd->vthis->ir.irField->index, ".vthis"));
                }
                if (depth > 1) {
                    src = DtoBitCast(src, getVoidPtrType());
                    LLValue* dst = DtoBitCast(frame, getVoidPtrType());
                    DtoMemCpy(dst, src, DtoConstSize_t((depth-1) * PTRSIZE),
                        getABITypeAlign(getVoidPtrType()));
                }
                // Copy nestArg into framelist; the outer frame is not in the list of pointers
                src = DtoBitCast(src, types[depth-1]);
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
                    DtoAlignedStore(vd->ir.irLocal->value, gep);
                    vd->ir.irLocal->byref = true;
                } else if (vd->isRef() || vd->isOut()) {
                    // This slot is initialized in DtoNestedInit, to handle things like byref foreach variables
                    // which move around in memory.
                    vd->ir.irLocal->byref = true;
                } else {
                    Logger::println("nested var:   %s", vd->toChars());
                    if (vd->ir.irLocal->value)
                        Logger::cout() << "Pre-existing value: " << *vd->ir.irLocal->value << '\n';
                    assert(!vd->ir.irLocal->value);
                    vd->ir.irLocal->value = gep;
                    vd->ir.irLocal->byref = false;
                }
            }
        } else if (FuncDeclaration* parFunc = getParentFunc(fd, true)) {
            // Propagate context arg properties if the context arg is passed on unmodified.
            fd->ir.irFunc->frameType = parFunc->ir.irFunc->frameType;
            fd->ir.irFunc->depth = parFunc->ir.irFunc->depth;
        }
    }
    else {
        assert(0 && "Not implemented yet");
    }
}
