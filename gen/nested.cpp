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
    NCStruct,
    
    /// Context is an array of pointers to nested contexts. Each function with variables
    /// accessed by nested functions puts them in a struct, and appends a pointer to that
    /// struct to the array.
    // FIXME: implement
    NCHybrid
};

static cl::opt<NestedCtxType> nestedCtx("nested-ctx",
    cl::desc("How to construct a nested function's context:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(NCArray,  "array",  "Array of pointers to variables (including multi-level)"),
        //clEnumValN(NCStruct, "struct", "Struct of variables (with multi-level via linked list)"),
        //clEnumValN(NCHybrid, "hybrid", "Array of pointers to structs of variables"),
        clEnumValEnd),
    cl::init(NCArray));


/****************************************************************************************/
/*////////////////////////////////////////////////////////////////////////////////////////
// NESTED VARIABLE HELPERS
////////////////////////////////////////////////////////////////////////////////////////*/

DValue* DtoNestedVariable(Loc loc, Type* astype, VarDeclaration* vd)
{
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
        val = DtoLoad(val);
        assert(vd->ir.irLocal->value);
        val = DtoBitCast(val, vd->ir.irLocal->value->getType(), vd->toChars());
        return new DVarValue(astype, vd, val);
    }
    else {
        assert(0 && "Not implemented yet");
    }
}

void DtoNestedInit(VarDeclaration* vd)
{
    if (nestedCtx == NCArray) {
        // alloca as usual if no value already
        if (!vd->ir.irLocal->value)
            vd->ir.irLocal->value = DtoAlloca(DtoType(vd->type), vd->toChars());
        
        // store the address into the nested vars array
        assert(vd->ir.irLocal->nestedIndex >= 0);
        LLValue* gep = DtoGEPi(gIR->func()->decl->ir.irFunc->nestedVar, 0, vd->ir.irLocal->nestedIndex);
        
        assert(isaPointer(vd->ir.irLocal->value));
        LLValue* val = DtoBitCast(vd->ir.irLocal->value, getVoidPtrType());
        
        DtoStore(val, gep);
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
                DtoMemCpy(nestedVars, src, DtoConstSize_t(nparelems*PTRSIZE));
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
                    DtoStore(val, gep);
                }
                else
                {
                    Logger::println("nested var:   %s", vd->toChars());
                }

                vd->ir.irLocal->nestedIndex = idx++;
            }

            // fixup nested result variable
        #if DMDV2
            if (fd->vresult && fd->vresult->nestedrefs.dim)
        #else
            if (fd->vresult && fd->vresult->nestedref)
        #endif
            {
                Logger::println("nested vresult value: %s", fd->vresult->toChars());
                LLValue* gep = DtoGEPi(nestedVars, 0, fd->vresult->ir.irLocal->nestedIndex);
                LLValue* val = DtoBitCast(fd->vresult->ir.irLocal->value, getVoidPtrType());
                DtoStore(val, gep);
            }
        }
    }
    else {
        assert(0 && "Not implemented yet");
    }
}
