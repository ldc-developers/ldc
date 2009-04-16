#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/classes.h"
#include "gen/structs.h"
#include "gen/functions.h"
#include "gen/runtime.h"
#include "gen/dvalue.h"
#include "gen/nested.h"
#include "gen/utils.h"

#include "ir/irstruct.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoResolveClass(ClassDeclaration* cd)
{
    // make sure the base classes are processed first
    if (cd->baseClass)
        cd->baseClass->codegen(Type::sir);

    if (cd->ir.resolved) return;
    cd->ir.resolved = true;

    Logger::println("DtoResolveClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // make sure type exists
    DtoType(cd->type);

    // create IrStruct
    assert(cd->ir.irStruct == NULL);
    IrStruct* irstruct = new IrStruct(cd);
    cd->ir.irStruct = irstruct;

    bool needs_def = mustDefineSymbol(cd);

    // emit the ClassZ symbol
    LLGlobalVariable* ClassZ = irstruct->getClassInfoSymbol();

    // interface only emit typeinfo and classinfo
    if (!cd->isInterfaceDeclaration())
    {
        // emit the initZ symbol
        LLGlobalVariable* initZ = irstruct->getInitSymbol();
        // emit the vtblZ symbol
        LLGlobalVariable* vtblZ = irstruct->getVtblSymbol();

        // emit the interfaceInfosZ symbol if necessary
        if (cd->vtblInterfaces && cd->vtblInterfaces->dim > 0)
            irstruct->getInterfaceArraySymbol(); // initializer is applied when it's built

        // perform definition
        if (needs_def)
        {
            // set symbol initializers
            initZ->setInitializer(irstruct->getDefaultInit());
            vtblZ->setInitializer(irstruct->getVtblInit());
        }

        // emit members
        if (cd->members)
        {
            ArrayIter<Dsymbol> it(*cd->members);
            while (!it.done())
            {
                Dsymbol* member = it.get();
                if (member)
                    member->codegen(Type::sir);
                it.next();
            }
        }
    }

    // emit typeinfo
    DtoTypeInfoOf(cd->type);

    // define classinfo
    if (needs_def)
    {
        ClassZ->setInitializer(irstruct->getClassInfoInit());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoNewClass(Loc loc, TypeClass* tc, NewExp* newexp)
{
    // resolve type
    tc->sym->codegen(Type::sir);

    // allocate
    LLValue* mem;
    if (newexp->onstack)
    {
        mem = DtoAlloca(DtoType(tc)->getContainedType(0), ".newclass_alloca");
    }
    // custom allocator
    else if (newexp->allocator)
    {
        newexp->allocator->codegen(Type::sir);
        DFuncValue dfn(newexp->allocator, newexp->allocator->ir.irFunc->func);
        DValue* res = DtoCallFunction(newexp->loc, NULL, &dfn, newexp->newargs);
        mem = DtoBitCast(res->getRVal(), DtoType(tc), ".newclass_custom");
    }
    // default allocator
    else
    {
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocclass");
        LLConstant* ci = DtoBitCast(tc->sym->ir.irStruct->getClassInfoSymbol(), DtoType(ClassDeclaration::classinfo->type));
        mem = gIR->CreateCallOrInvoke(fn, ci, ".newclass_gc_alloc").getInstruction();
        mem = DtoBitCast(mem, DtoType(tc), ".newclass_gc");
    }

    // init
    DtoInitClass(tc, mem);

    // init inner-class outer reference
    if (newexp->thisexp)
    {
        Logger::println("Resolving outer class");
        LOG_SCOPE;
        DValue* thisval = newexp->thisexp->toElem(gIR);
        size_t idx = tc->sym->vthis->ir.irField->index;
        LLValue* src = thisval->getRVal();
        LLValue* dst = DtoGEPi(mem,0,idx,"tmp");
        if (Logger::enabled())
            Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
        DtoStore(src, dst);
    }
    // set the context for nested classes
    else if (tc->sym->isNested() && tc->sym->vthis)
    {
        Logger::println("Resolving nested context");
        LOG_SCOPE;

        // get context
        LLValue* nest = DtoNestedContext(loc, tc->sym);

        // store into right location
        size_t idx = tc->sym->vthis->ir.irField->index;
        LLValue* gep = DtoGEPi(mem,0,idx,"tmp");
        DtoStore(DtoBitCast(nest, gep->getType()->getContainedType(0)), gep);
    }

    // call constructor
    if (newexp->member)
    {
        Logger::println("Calling constructor");
        assert(newexp->arguments != NULL);
        newexp->member->codegen(Type::sir);
        DFuncValue dfn(newexp->member, newexp->member->ir.irFunc->func, mem);
        return DtoCallFunction(newexp->loc, tc, &dfn, newexp->arguments);
    }

    // return default constructed class
    return new DImValue(tc, mem);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass* tc, LLValue* dst)
{
    tc->sym->codegen(Type::sir);

    size_t presz = 2*getTypePaddedSize(DtoSize_t());
    uint64_t n = getTypePaddedSize(tc->ir.type->get()) - presz;

    // set vtable field seperately, this might give better optimization
    LLValue* tmp = DtoGEPi(dst,0,0,"vtbl");
    LLValue* val = DtoBitCast(tc->sym->ir.irStruct->getVtblSymbol(), tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // monitor always defaults to zero
    tmp = DtoGEPi(dst,0,1,"monitor");
    val = llvm::Constant::getNullValue(tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // done?
    if (n == 0)
        return;

    // copy the rest from the static initializer
    LLValue* dstarr = DtoGEPi(dst,0,2,"tmp");
    LLValue* srcarr = DtoGEPi(tc->sym->ir.irStruct->getInitSymbol(),0,2,"tmp");

    DtoMemCpy(dstarr, srcarr, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_callfinalizer");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end(), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastClass(DValue* val, Type* _to)
{
    Logger::println("DtoCastClass(%s, %s)", val->getType()->toChars(), _to->toChars());
    LOG_SCOPE;

    Type* to = _to->toBasetype();

    // class -> pointer
    if (to->ty == Tpointer) {
        Logger::println("to pointer");
        const LLType* tolltype = DtoType(_to);
        LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
        return new DImValue(_to, rval);
    }
    // class -> bool
    else if (to->ty == Tbool) {
        Logger::println("to bool");
        LLValue* llval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(llval->getType());
        return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero, "tmp"));
    }

    // must be class/interface
    assert(to->ty == Tclass);
    TypeClass* tc = (TypeClass*)to;

    // from type
    Type* from = val->getType()->toBasetype();
    TypeClass* fc = (TypeClass*)from;

    // x -> interface
    if (InterfaceDeclaration* it = tc->sym->isInterfaceDeclaration()) {
        Logger::println("to interface");
        // interface -> interface
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("from interface");
            return DtoDynamicCastInterface(val, _to);
        }
        // class -> interface - static cast
        else if (it->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");

            // get the from class
            ClassDeclaration* cd = fc->sym->isClassDeclaration();
            DtoResolveClass(cd); // add this
            IrStruct* irstruct = cd->ir.irStruct;
            IrTypeClass* typeclass = fc->irtype->isClass();

            // find interface impl
            
            size_t i_index = typeclass->getInterfaceIndex(it);
            assert(i_index != ~0 && "requesting interface that is not implemented by this class");

            // offset pointer
            LLValue* v = val->getRVal();
            LLValue* orig = v;
            v = DtoGEPi(v, 0, i_index);
            const LLType* ifType = DtoType(_to);
            if (Logger::enabled())
            {
                Logger::cout() << "V = " << *v << std::endl;
                Logger::cout() << "T = " << *ifType << std::endl;
            }
            v = DtoBitCast(v, ifType);

            // Check whether the original value was null, and return null if so.
            // Sure we could have jumped over the code above in this case, but
            // it's just a GEP and (maybe) a pointer-to-pointer BitCast, so it
            // should be pretty cheap and perfectly safe even if the original was null.
            LLValue* isNull = gIR->ir->CreateICmpEQ(orig, LLConstant::getNullValue(orig->getType()), ".nullcheck");
            v = gIR->ir->CreateSelect(isNull, LLConstant::getNullValue(ifType), v, ".interface");

            // return r-value
            return new DImValue(_to, v);
        }
        // class -> interface
        else {
            Logger::println("from object");
            return DtoDynamicCastObject(val, _to);
        }
    }
    // x -> class
    else {
        Logger::println("to class");
        int poffset;
        // interface -> class
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("interface cast");
            return DtoDynamicCastInterface(val, _to);
        }
        // class -> class - static down cast
        else if (tc->sym->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");
            const LLType* tolltype = DtoType(_to);
            LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
            return new DImValue(_to, rval);
        }
        // class -> class - dynamic up cast
        else {
            Logger::println("dynamic up cast");
            return DtoDynamicCastObject(val, _to);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastObject(DValue* val, Type* _to)
{
    // call:
    // Object _d_dynamic_cast(Object o, ClassInfo c)

    ClassDeclaration::object->codegen(Type::sir);
    ClassDeclaration::classinfo->codegen(Type::sir);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_dynamic_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // Object o
    LLValue* obj = val->getRVal();
    obj = DtoBitCast(obj, funcTy->getParamType(0));
    assert(funcTy->getParamType(0) == obj->getType());

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    to->sym->codegen(Type::sir);

    LLValue* cinfo = to->sym->ir.irStruct->getClassInfoSymbol();
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));
    assert(funcTy->getParamType(1) == cinfo->getType());

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, obj, cinfo, "tmp").getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastInterfaceToObject(DValue* val, Type* to)
{
    // call:
    // Object _d_toObject(void* p)

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_toObject");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // void* p
    LLValue* tmp = val->getRVal();
    tmp = DtoBitCast(tmp, funcTy->getParamType(0));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke(func, tmp, "tmp").getInstruction();

    // cast return value
    if (to != NULL)
        ret = DtoBitCast(ret, DtoType(to));
    else
        to = ClassDeclaration::object->type;

    return new DImValue(to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastInterface(DValue* val, Type* _to)
{
    // call:
    // Object _d_interface_cast(void* p, ClassInfo c)

    ClassDeclaration::object->codegen(Type::sir);
    ClassDeclaration::classinfo->codegen(Type::sir);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_interface_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // void* p
    LLValue* ptr = val->getRVal();
    ptr = DtoBitCast(ptr, funcTy->getParamType(0));

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    to->sym->codegen(Type::sir);
    LLValue* cinfo = to->sym->ir.irStruct->getClassInfoSymbol();
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, ptr, cinfo, "tmp").getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoIndexClass(LLValue* src, ClassDeclaration* cd, VarDeclaration* vd)
{
    Logger::println("indexing class field %s:", vd->toPrettyChars());
    LOG_SCOPE;

    if (Logger::enabled())
        Logger::cout() << "src: " << *src << '\n';

    // make sure class is resolved
    DtoResolveClass(cd);

    // vd must be a field
    IrField* field = vd->ir.irField;
    assert(field);

    // get the start pointer
    const LLType* st = DtoType(cd->type);
    // cast to the struct type
    src = DtoBitCast(src, st);

    // gep to the index
    if (Logger::enabled())
    {
        Logger::cout() << "src2: " << *src << '\n';
        Logger::cout() << "index: " << field->index << '\n';
        Logger::cout() << "srctype: " << *src->getType() << '\n';
    }
    LLValue* val = DtoGEPi(src, 0, field->index);

    // do we need to offset further? (union area)
    if (field->unionOffset)
    {
        // cast to void*
        val = DtoBitCast(val, getVoidPtrType());
        // offset
        val = DtoGEPi1(val, field->unionOffset);
    }

    // cast it to the right type
    val = DtoBitCast(val, getPtrToType(DtoType(vd->type)));

    if (Logger::enabled())
        Logger::cout() << "value: " << *val << '\n';

    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl)
{
    // sanity checks
    assert(fdecl->isVirtual());
    assert(!fdecl->isFinal());
    assert(fdecl->vtblIndex > 0); // 0 is always ClassInfo/Interface*
    assert(inst->getType()->toBasetype()->ty == Tclass);

    // get instance
    LLValue* vthis = inst->getRVal();
    if (Logger::enabled())
        Logger::cout() << "vthis: " << *vthis << '\n';

    LLValue* funcval = vthis;
    // get the vtbl for objects
    if (!fdecl->isMember()->isInterfaceDeclaration())
        funcval = DtoGEPi(funcval, 0, 0, "tmp");
    // load vtbl ptr
    funcval = DtoLoad(funcval);
    // index vtbl
    funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, fdecl->toChars());
    // load funcptr
    funcval = DtoAlignedLoad(funcval);

    if (Logger::enabled())
        Logger::cout() << "funcval: " << *funcval << '\n';

    // cast to final funcptr type
    funcval = DtoBitCast(funcval, getPtrToType(DtoType(fdecl->type)));
    if (Logger::enabled())
        Logger::cout() << "funcval casted: " << *funcval << '\n';

    return funcval;
}

//////////////////////////////////////////////////////////////////////////////////////////

#if GENERATE_OFFTI

// build a single element for the OffsetInfo[] of ClassInfo
static LLConstant* build_offti_entry(ClassDeclaration* cd, VarDeclaration* vd)
{
    std::vector<LLConstant*> inits(2);

    // size_t offset;
    //
    assert(vd->ir.irField);
    // grab the offset from llvm and the formal class type
    size_t offset = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(vd->ir.irField->index);
    // offset nested struct/union fields
    offset += vd->ir.irField->unionOffset;

    // assert that it matches DMD
    Logger::println("offsets: %lu vs %u", offset, vd->offset);
    assert(offset == vd->offset);

    inits[0] = DtoConstSize_t(offset);

    // TypeInfo ti;
    inits[1] = DtoTypeInfoOf(vd->type, true);

    // done
    return llvm::ConstantStruct::get(inits);
}

static LLConstant* build_offti_array(ClassDeclaration* cd, const LLType* arrayT)
{
    IrStruct* irstruct = cd->ir.irStruct;

    size_t nvars = irstruct->varDecls.size();
    std::vector<LLConstant*> arrayInits(nvars);

    for (size_t i=0; i<nvars; i++)
    {
        arrayInits[i] = build_offti_entry(cd, irstruct->varDecls[i]);
    }

    LLConstant* size = DtoConstSize_t(nvars);
    LLConstant* ptr;

    if (nvars == 0)
        return LLConstant::getNullValue( arrayT );

    // array type
    const llvm::ArrayType* arrTy = llvm::ArrayType::get(arrayInits[0]->getType(), nvars);
    LLConstant* arrInit = llvm::ConstantArray::get(arrTy, arrayInits);

    // mangle
    std::string name(cd->type->vtinfo->toChars());
    name.append("__OffsetTypeInfos");

    // create symbol
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrTy,true,DtoInternalLinkage(cd),arrInit,name,gIR->module);
    ptr = DtoBitCast(gvar, getPtrToType(arrTy->getElementType()));

    return DtoConstSlice(size, ptr);
}

#endif // GENERATE_OFFTI

static LLConstant* build_class_dtor(ClassDeclaration* cd)
{
    FuncDeclaration* dtor = cd->dtor;

    // if no destructor emit a null
    if (!dtor)
        return getNullPtr(getVoidPtrType());

    dtor->codegen(Type::sir);
    return llvm::ConstantExpr::getBitCast(dtor->ir.irFunc->func, getPtrToType(LLType::Int8Ty));
}

static unsigned build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code
    unsigned flags = 0;
    //flags |= isCOMclass(); // IUnknown
    bool hasOffTi = false;
    if (cd->ctor) flags |= 8;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        if (sm->isVarDeclaration() && !sm->isVarDeclaration()->isDataseg()) // is this enough?
            hasOffTi = true;
        //printf("sm = %s %s\n", sm->kind(), sm->toChars());
        if (sm->hasPointers())
            goto L2;
        }
    }
    }
    flags |= 2;         // no pointers
L2:
    if (hasOffTi)
        flags |= 4;
    return flags;
}

LLConstant* DtoDefineClassInfo(ClassDeclaration* cd)
{
//     The layout is:
//        {
//         void **vptr;
//         monitor_t monitor;
//         byte[] initializer;     // static initialization data
//         char[] name;        // class name
//         void *[] vtbl;
//         Interface[] interfaces;
//         ClassInfo *base;        // base class
//         void *destructor;
//         void *invariant;        // class invariant
//         uint flags;
//         void *deallocator;
//         OffsetTypeInfo[] offTi;
//         void *defaultConstructor;
//        }

    Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    IrStruct* ir = cd->ir.irStruct;

    assert(cd->type->ty == Tclass);

    TypeClass* cdty = (TypeClass*)cd->type;

    // holds the list of initializers for llvm
    std::vector<LLConstant*> inits;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    cinfo->codegen(Type::sir);

    LLConstant* c;

    const LLType* voidPtr = getVoidPtrType();
    const LLType* voidPtrPtr = getPtrToType(voidPtr);

    // own vtable
    c = cinfo->ir.irStruct->getVtblSymbol();
    inits.push_back(c);

    // monitor
    c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

    // byte[] init
    if (cd->isInterfaceDeclaration())
        c = DtoConstSlice(DtoConstSize_t(0), LLConstant::getNullValue(voidPtr));
    else
    {
        c = DtoBitCast(ir->getInitSymbol(), voidPtr);
        //Logger::cout() << *ir->constInit->getType() << std::endl;
        size_t initsz = getTypePaddedSize(ir->getInitSymbol()->getType()->getContainedType(0));
        c = DtoConstSlice(DtoConstSize_t(initsz), c);
    }
    inits.push_back(c);

    // class name
    // from dmd
    char *name = cd->ident->toChars();
    size_t namelen = strlen(name);
    if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0))
    {
        name = cd->toPrettyChars();
        namelen = strlen(name);
    }
    c = DtoConstString(name);
    inits.push_back(c);

    // vtbl array
    if (cd->isInterfaceDeclaration())
        c = DtoConstSlice(DtoConstSize_t(0), LLConstant::getNullValue(getPtrToType(voidPtr)));
    else {
        c = DtoBitCast(ir->getVtblSymbol(), voidPtrPtr);
        c = DtoConstSlice(DtoConstSize_t(cd->vtbl.dim), c);
    }
    inits.push_back(c);

    // interfaces array
    c = ir->getClassInfoInterfaces();
    inits.push_back(c);

    // base classinfo
    // interfaces never get a base , just the interfaces[]
    if (cd->baseClass && !cd->isInterfaceDeclaration()) {
        c = cd->baseClass->ir.irStruct->getClassInfoSymbol();
        assert(c);
        inits.push_back(c);
    }
    else {
        // null
        c = LLConstant::getNullValue(DtoType(cinfo->type));
        inits.push_back(c);
    }

    // destructor
    if (cd->isInterfaceDeclaration())
        c = LLConstant::getNullValue(voidPtr);
    else
        c = build_class_dtor(cd);
    inits.push_back(c);

    // invariant
    VarDeclaration* invVar = (VarDeclaration*)cinfo->fields.data[6];
    const LLType* invTy = DtoType(invVar->type);
    if (cd->inv)
    {
        cd->inv->codegen(Type::sir);
        c = cd->inv->ir.irFunc->func;
        c = DtoBitCast(c, invTy);
    }
    else
        c = LLConstant::getNullValue(invTy);
    inits.push_back(c);

    // uint flags
    if (cd->isInterfaceDeclaration())
        c = DtoConstUint(0);
    else {
        unsigned flags = build_classinfo_flags(cd);
        c = DtoConstUint(flags);
    }
    inits.push_back(c);

    // deallocator
    if (cd->aggDelete)
    {
        cd->aggDelete->codegen(Type::sir);
        c = cd->aggDelete->ir.irFunc->func;
        c = DtoBitCast(c, voidPtr);
    }
    else
        c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

    // offset typeinfo
    VarDeclaration* offTiVar = (VarDeclaration*)cinfo->fields.data[9];
    const LLType* offTiTy = DtoType(offTiVar->type);

#if GENERATE_OFFTI

    if (cd->isInterfaceDeclaration())
        c = LLConstant::getNullValue(offTiTy);
    else
        c = build_offti_array(cd, offTiTy);

#else // GENERATE_OFFTI

    c = LLConstant::getNullValue(offTiTy);

#endif // GENERATE_OFFTI

    inits.push_back(c);

    // default constructor
    if (cd->defaultCtor)
    {
        cd->defaultCtor->codegen(Type::sir);
        c = isaConstant(cd->defaultCtor->ir.irFunc->func);
        c = DtoBitCast(c, voidPtr);
    }
    else
        c = LLConstant::getNullValue(voidPtr);
    inits.push_back(c);

#if DMDV2

    // xgetMembers
    VarDeclaration* xgetVar = (VarDeclaration*)cinfo->fields.data[11];
    const LLType* xgetTy = DtoType(xgetVar->type);

    // FIXME: fill it out!
    inits.push_back( LLConstant::getNullValue(xgetTy) );
#endif

    /*size_t n = inits.size();
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
    }*/

    // build the initializer
    LLConstant* finalinit = llvm::ConstantStruct::get(inits);
    //Logger::cout() << "built the classinfo initializer:\n" << *finalinit <<'\n';
    ir->constClassInfo = finalinit;

    // sanity check
    assert(finalinit->getType() == ir->classInfo->getType()->getContainedType(0) &&
        "__ClassZ initializer does not match the ClassInfo type");
    

    // return initializer
    return finalinit;
}
