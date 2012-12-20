//===-- classes.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/dvalue.h"
#include "gen/irstate.h"

#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/utils.h"

#include "ir/irstruct.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoResolveClass(ClassDeclaration* cd)
{
    // make sure the base classes are processed first
    ArrayIter<BaseClass> base_iter(cd->baseclasses);
    while (base_iter.more())
    {
        BaseClass* bc = base_iter.get();
        if (bc)
        {
            bc->base->codegen(Type::sir);
        }
        base_iter.next();
    }

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

    // make sure all fields really get their ir field
    ArrayIter<VarDeclaration> it(cd->fields);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        if (vd->ir.irField == NULL) {
            new IrField(vd);
        } else {
            IF_LOG Logger::println("class field already exists!!!");
        }
    }

    bool needs_def = mustDefineSymbol(cd);

    // emit the ClassZ symbol
    LLGlobalVariable* ClassZ = irstruct->getClassInfoSymbol();

    // emit the interfaceInfosZ symbol if necessary
    if (cd->vtblInterfaces && cd->vtblInterfaces->dim > 0)
        irstruct->getInterfaceArraySymbol(); // initializer is applied when it's built

    // interface only emit typeinfo and classinfo
    if (cd->isInterfaceDeclaration())
    {
        irstruct->initializeInterface();
    }
    else
    {
        // emit the initZ symbol
        LLGlobalVariable* initZ = irstruct->getInitSymbol();
        // emit the vtblZ symbol
        LLGlobalVariable* vtblZ = irstruct->getVtblSymbol();

        // perform definition
        if (needs_def)
        {
            // set symbol initializers
            initZ->setInitializer(irstruct->getDefaultInit());
            vtblZ->setInitializer(irstruct->getVtblInit());
        }
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

    if (needs_def)
    {
        // emit typeinfo
        DtoTypeInfoOf(cd->type);

        // define classinfo
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
        // FIXME align scope class to its largest member
        mem = DtoRawAlloca(DtoType(tc)->getContainedType(0), 0, ".newclass_alloca");
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
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, _d_allocclass);
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
        DtoStore(src, DtoBitCast(dst, getPtrToType(src->getType())));
    }
    // set the context for nested classes
    else if (tc->sym->isNested() && tc->sym->vthis)
    {
        DtoResolveNestedContext(loc, tc->sym, mem);
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

    uint64_t n = tc->sym->structsize - PTRSIZE * 2;

    // set vtable field seperately, this might give better optimization
    LLValue* tmp = DtoGEPi(dst,0,0,"vtbl");
    LLValue* val = DtoBitCast(tc->sym->ir.irStruct->getVtblSymbol(), tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // monitor always defaults to zero
    tmp = DtoGEPi(dst,0,1,"monitor");
    val = LLConstant::getNullValue(tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // done?
    if (n == 0)
        return;

    // copy the rest from the static initializer
    LLValue* dstarr = DtoGEPi(dst,0,2,"tmp");

    // init symbols might not have valid types
    LLValue* initsym = tc->sym->ir.irStruct->getInitSymbol();
    initsym = DtoBitCast(initsym, DtoType(tc));
    LLValue* srcarr = DtoGEPi(initsym,0,2,"tmp");

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
    gIR->CreateCallOrInvoke(fn, arg, "");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastClass(DValue* val, Type* _to)
{
    Logger::println("DtoCastClass(%s, %s)", val->getType()->toChars(), _to->toChars());
    LOG_SCOPE;

    Type* to = _to->toBasetype();

    // class -> pointer
    if (to->ty == Tpointer) {
        IF_LOG Logger::println("to pointer");
        LLType* tolltype = DtoType(_to);
        LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
        return new DImValue(_to, rval);
    }
    // class -> bool
    else if (to->ty == Tbool) {
        IF_LOG Logger::println("to bool");
        LLValue* llval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(llval->getType());
        return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero, "tmp"));
    }
    // class -> integer
    else if (to->isintegral()) {
        IF_LOG Logger::println("to %s", to->toChars());

        // get class ptr
        LLValue* v = val->getRVal();
        // cast to size_t
        v = gIR->ir->CreatePtrToInt(v, DtoSize_t(), "");
        // cast to the final int type
        DImValue im(Type::tsize_t, v);
        Loc loc;
        return DtoCastInt(loc, &im, _to);
    }

    // must be class/interface
    assert(to->ty == Tclass);
    TypeClass* tc = static_cast<TypeClass*>(to);

    // from type
    Type* from = val->getType()->toBasetype();
    TypeClass* fc = static_cast<TypeClass*>(from);

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
            IrTypeClass* typeclass = stripModifiers(fc)->irtype->isClass();

            // find interface impl

            size_t i_index = typeclass->getInterfaceIndex(it);
            assert(i_index != ~0UL && "requesting interface that is not implemented by this class");

            // offset pointer
            LLValue* v = val->getRVal();
            LLValue* orig = v;
            v = DtoGEPi(v, 0, i_index);
            LLType* ifType = DtoType(_to);
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
        // interface -> class
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("interface cast");
            return DtoDynamicCastInterface(val, _to);
        }
        // class -> class - static down cast
        else if (tc->sym->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");
            LLType* tolltype = DtoType(_to);
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
    LLFunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // Object o
    LLValue* obj = val->getRVal();
    obj = DtoBitCast(obj, funcTy->getParamType(0));
    assert(funcTy->getParamType(0) == obj->getType());

    // ClassInfo c
    TypeClass* to = static_cast<TypeClass*>(_to->toBasetype());
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
    LLFunctionType* funcTy = func->getFunctionType();

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
    LLFunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // void* p
    LLValue* ptr = val->getRVal();
    ptr = DtoBitCast(ptr, funcTy->getParamType(0));

    // ClassInfo c
    TypeClass* to = static_cast<TypeClass*>(_to->toBasetype());
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
    LLType* st = DtoType(cd->type);
    // cast to the struct type
    src = DtoBitCast(src, st);

    // gep to the index
#if 0
    if (Logger::enabled())
    {
        Logger::cout() << "src2: " << *src << '\n';
        Logger::cout() << "index: " << field->index << '\n';
        Logger::cout() << "srctype: " << *src->getType() << '\n';
    }
#endif
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

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl, char* name)
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
    funcval = DtoGEPi(funcval, 0, 0, "tmp");
    // load vtbl ptr
    funcval = DtoLoad(funcval);
    // index vtbl
    std::string vtblname = name;
    vtblname.append("@vtbl");
    funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, vtblname.c_str());
    // load funcptr
    funcval = DtoAlignedLoad(funcval);

    if (Logger::enabled())
        Logger::cout() << "funcval: " << *funcval << '\n';

    // cast to final funcptr type
    funcval = DtoBitCast(funcval, getPtrToType(DtoType(fdecl->type)));

    // postpone naming until after casting to get the name in call instructions
    funcval->setName(name);

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
    size_t offset = gDataLayout->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(vd->ir.irField->index);
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

static LLConstant* build_offti_array(ClassDeclaration* cd, LLType* arrayT)
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
    LLArrayType* arrTy = llvm::ArrayType::get(arrayInits[0]->getType(), nvars);
    LLConstant* arrInit = LLConstantArray::get(arrTy, arrayInits);

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
    return llvm::ConstantExpr::getBitCast(dtor->ir.irFunc->func, getPtrToType(LLType::getInt8Ty(gIR->context())));
}

static unsigned build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code
    unsigned flags = 0;
    flags |= cd->isCOMclass(); // IUnknown
    bool hasOffTi = false;
    if (cd->ctor)
        flags |= 8;
    if (cd->isabstract)
        flags |= 64;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
        if (!cd2->members)
            continue;
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
            Dsymbol *sm = static_cast<Dsymbol *>(cd2->members->data[i]);
            if (sm->isVarDeclaration() && !sm->isVarDeclaration()->isDataseg()) // is this enough?
                hasOffTi = true;
            //printf("sm = %s %s\n", sm->kind(), sm->toChars());
            if (sm->hasPointers())
                goto L2;
        }
    }
    flags |= 2;         // no pointers
L2:
    if (hasOffTi)
        flags |= 4;

    // always define the typeinfo field.
    // why would ever not do this?
    flags |= 32;

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
//         version(D_Version2)
//              immutable(void)* m_RTInfo;
//         else
//              TypeInfo typeinfo; // since dmd 1.045
//        }

    Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    TypeClass* cdty = static_cast<TypeClass*>(cd->type);

    IrStruct* ir = cd->ir.irStruct;
    assert(ir);

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;

    if (cinfo->fields.dim != 12)
    {
        error("object.d ClassInfo class is incorrect");
        fatal();
    }

    // use the rtti builder
    RTTIBuilder b(ClassDeclaration::classinfo);

    LLConstant* c;

    LLType* voidPtr = getVoidPtrType();
    LLType* voidPtrPtr = getPtrToType(voidPtr);

    // byte[] init
    if (cd->isInterfaceDeclaration())
    {
        b.push_null_void_array();
    }
    else
    {
        LLType* cd_type = cdty->irtype->isClass()->getMemoryLLType();
        size_t initsz = getTypePaddedSize(cd_type);
        b.push_void_array(initsz, ir->getInitSymbol());
    }

    // class name
    // code from dmd
    const char *name = cd->ident->toChars();
    size_t namelen = strlen(name);
    if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0))
    {
        name = cd->toPrettyChars();
        namelen = strlen(name);
    }
    b.push_string(name);

    // vtbl array
    if (cd->isInterfaceDeclaration())
    {
        b.push_array(0, getNullValue(voidPtrPtr));
    }
    else
    {
        c = DtoBitCast(ir->getVtblSymbol(), voidPtrPtr);
        b.push_array(cd->vtbl.dim, c);
    }

    // interfaces array
    b.push(ir->getClassInfoInterfaces());

    // base classinfo
    // interfaces never get a base , just the interfaces[]
    if (cd->baseClass && !cd->isInterfaceDeclaration())
        b.push_classinfo(cd->baseClass);
    else
        b.push_null(cinfo->type);

    // destructor
    if (cd->isInterfaceDeclaration())
        b.push_null_vp();
    else
        b.push(build_class_dtor(cd));

    // invariant
    VarDeclaration* invVar = static_cast<VarDeclaration*>(cinfo->fields.data[6]);
    b.push_funcptr(cd->inv, invVar->type);

    // uint flags
    unsigned flags;
    if (cd->isInterfaceDeclaration())
        flags = 4 | cd->isCOMinterface() | 32;
    else
        flags = build_classinfo_flags(cd);
    b.push_uint(flags);

    // deallocator
    b.push_funcptr(cd->aggDelete, Type::tvoid->pointerTo());

    // offset typeinfo
    VarDeclaration* offTiVar = static_cast<VarDeclaration*>(cinfo->fields.data[9]);

#if GENERATE_OFFTI

    if (cd->isInterfaceDeclaration())
        b.push_null(offTiVar->type);
    else
        b.push(build_offti_array(cd, DtoType(offTiVar->type)));

#else // GENERATE_OFFTI

    b.push_null(offTiVar->type);

#endif // GENERATE_OFFTI

    // default constructor
    VarDeclaration* defConstructorVar = static_cast<VarDeclaration*>(cinfo->fields.data[10]);
    b.push_funcptr(cd->defaultCtor, defConstructorVar->type);

#if DMDV2
    // immutable(void)* m_RTInfo;
    // The cases where getRTInfo is null are not quite here, but the code is
    // modelled after what DMD does.
    if (cd->getRTInfo)
        b.push(cd->getRTInfo->toConstElem(gIR));
    else if (flags & 2)
        b.push_size_as_vp(0);       // no pointers
    else
        b.push_size_as_vp(1);       // has pointers
#else
    // typeinfo - since 1.045
    b.push_typeinfo(cd->type);
#endif

    /*size_t n = inits.size();
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
    }*/

    // build the initializer
    LLType *initType = ir->classInfo->getType()->getContainedType(0);
    LLConstant* finalinit = b.get_constant(isaStruct(initType));

    //Logger::cout() << "built the classinfo initializer:\n" << *finalinit <<'\n';
    ir->constClassInfo = finalinit;

    // sanity check
    assert(finalinit->getType() == initType &&
        "__ClassZ initializer does not match the ClassInfo type");

    // return initializer
    return finalinit;
}
