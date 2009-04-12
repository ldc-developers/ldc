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
    
    /// Context is a list of pointers to structs. Each function with variables
    /// accessed by nested functions puts them in a struct, and appends a
    /// pointer to that struct to it's local copy of the list.
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

static FuncDeclaration* getParentFunc(Dsymbol* sym) {
    Dsymbol* parent = sym->parent;
    assert(parent);
    while (parent && !parent->isFuncDeclaration())
        parent = parent->parent;
    
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
        FuncDeclaration *parentfunc = getParentFunc(vd);
        assert(parentfunc && "No parent function for nested variable?");
        
        LLValue* val = DtoBitCast(ctx, LLPointerType::getUnqual(parentfunc->ir.irFunc->framesType));
        val = DtoGEPi(val, 0, vd->ir.irLocal->nestedDepth);
        val = DtoAlignedLoad(val, (std::string(".frame.") + parentfunc->toChars()).c_str());
        val = DtoGEPi(val, 0, vd->ir.irLocal->nestedIndex, vd->toChars());
        if (vd->ir.irLocal->byref)
            val = DtoAlignedLoad(val);
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
    
    LLValue* nestedVar = gIR->func()->decl->ir.irFunc->nestedVar;
    
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
            Logger::println("Initializing non-parameter byref value");
            LLValue* framep = DtoGEPi(nestedVar, 0, vd->ir.irLocal->nestedDepth);
            
            FuncDeclaration *parentfunc = getParentFunc(vd);
            assert(parentfunc && "No parent function for nested variable?");
            LLValue* frame = DtoAlignedLoad(framep, (std::string(".frame.") + parentfunc->toChars()).c_str());
            
            LLValue* slot = DtoGEPi(frame, 0, vd->ir.irLocal->nestedIndex);
            DtoAlignedStore(vd->ir.irLocal->value, slot);
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

    // if this func has its own vars that are accessed by nested funcs
    // use its own context
    if (irfunc->nestedVar)
        return irfunc->nestedVar;
    // otherwise, it may have gotten a context from the caller
    else if (irfunc->nestArg)
        return irfunc->nestArg;
    // or just have a this argument
    else if (irfunc->thisArg)
    {
        ClassDeclaration* cd = irfunc->decl->isMember2()->isClassDeclaration();
        if (!cd || !cd->vthis)
            return getNullPtr(getVoidPtrType());
        LLValue* val = DtoLoad(irfunc->thisArg);
        return DtoLoad(DtoGEPi(val, 0,cd->vthis->ir.irField->index, ".vthis"));
    }
    else
    {
        return getNullPtr(getVoidPtrType());
    }
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
            typedef std::vector<const LLType*> TypeVec;
            TypeVec frametypes;
            if (!fd->isStatic()) {
                Dsymbol* par = fd->toParent2();
                while (par) {
                    if (FuncDeclaration* parfd = par->isFuncDeclaration()) {
                        // skip functions without nested parameters
                        if (!parfd->nestedVars.empty()) {
                            // Copy the types of parent function frames.
                            const LLStructType* parft = parfd->ir.irFunc->framesType;
                            frametypes.insert(frametypes.begin(), parft->element_begin(), parft->element_end());
                            break;  // That's all the info needed.
                        }
                    } else if (ClassDeclaration* parcd = par->isClassDeclaration()) {
                        // skip
                    } else {
                        break;
                    }
                    par = par->toParent2();
                }
            }
            unsigned depth = frametypes.size();
            
            // Construct a struct for the direct nested variables of this function, and update their indices to match.
            // TODO: optimize ordering for minimal space usage?
            TypeVec types;
            for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i)
            {
                VarDeclaration* vd = *i;
                if (!vd->ir.irLocal)
                    vd->ir.irLocal = new IrLocal(vd);
                
                vd->ir.irLocal->nestedDepth = depth;
                vd->ir.irLocal->nestedIndex = types.size();
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
            // Append current frame type to frame type list
            const LLType* frameType = LLStructType::get(types);
            frametypes.push_back(LLPointerType::getUnqual(frameType));
            
            // make struct type for nested frame list
            const LLStructType* nestedVarsTy = LLStructType::get(frametypes);
            
            // Store type in IrFunction
            IrFunction* irfunction = fd->ir.irFunc;
            irfunction->framesType = nestedVarsTy;
            
            // alloca it
            // FIXME: For D2, this should be a gc_malloc (or similar) call, not alloca
            LLValue* nestedVars = DtoAlloca(nestedVarsTy, ".frame_list");
            
            // copy parent frames into beginning
            if (depth != 0)
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
                    Logger::println("Indexing to 'this'");
                    src = DtoLoad(DtoGEPi(thisval, 0, cd->vthis->ir.irField->index, ".vthis"));
                }
                src = DtoBitCast(src, getVoidPtrType());
                LLValue* dst = DtoBitCast(nestedVars, getVoidPtrType());
                DtoMemCpy(dst, src, DtoConstSize_t(depth * PTRSIZE),
                    getABITypeAlign(getVoidPtrType()));
            }
            
            // Create frame for current function and append to frames list
            LLValue* frame = DtoAlloca(frameType, ".frame");
            // store current frame in list
            DtoAlignedStore(frame, DtoGEPi(nestedVars, 0, depth));
            
            // store context in IrFunction
            irfunction->nestedVar = nestedVars;
            
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
        }
    }
    else {
        assert(0 && "Not implemented yet");
    }
}
