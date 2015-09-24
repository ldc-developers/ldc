//===-- classes.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"
#include "mtype.h"
#include "target.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoResolveClass(ClassDeclaration* cd)
{
    if (cd->ir.isResolved()) return;
    cd->ir.setResolved();

    IF_LOG Logger::println("DtoResolveClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // make sure the base classes are processed first
    for (BaseClasses::iterator I = cd->baseclasses->begin(),
                               E = cd->baseclasses->end();
                               I != E; ++I)
    {
        DtoResolveClass((*I)->base);
    }

    // make sure type exists
    DtoType(cd->type);

    // create IrAggr
    IrAggr* irAggr = getIrAggr(cd, true);

    // make sure all fields really get their ir field
    for (VarDeclarations::iterator I = cd->fields.begin(),
                                   E = cd->fields.end();
                                   I != E; ++I)
    {
        VarDeclaration* vd = *I;
        IF_LOG {
            if (isIrFieldCreated(vd))
                Logger::println("class field already exists");
        }
        getIrField(vd, true);
    }

    // emit the interfaceInfosZ symbol if necessary
    if (cd->vtblInterfaces && cd->vtblInterfaces->dim > 0)
        irAggr->getInterfaceArraySymbol(); // initializer is applied when it's built

    // interface only emit typeinfo and classinfo
    if (cd->isInterfaceDeclaration())
    {
        irAggr->initializeInterface();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoNewClass(Loc& loc, TypeClass* tc, NewExp* newexp)
{
    // resolve type
    DtoResolveClass(tc->sym);

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
        DtoResolveFunction(newexp->allocator);
        DFuncValue dfn(newexp->allocator, getIrFunc(newexp->allocator)->func);
        DValue* res = DtoCallFunction(newexp->loc, NULL, &dfn, newexp->newargs);
        mem = DtoBitCast(res->getRVal(), DtoType(tc), ".newclass_custom");
    }
    // default allocator
    else
    {
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_newclass");
        LLConstant* ci = DtoBitCast(getIrAggr(tc->sym)->getClassInfoSymbol(), DtoType(Type::typeinfoclass->type));
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
        DValue* thisval = toElem(newexp->thisexp);
        unsigned idx = getFieldGEPIndex(tc->sym, tc->sym->vthis);
        LLValue* src = thisval->getRVal();
        LLValue* dst = DtoGEPi(mem, 0, idx);
        IF_LOG Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
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
        // evaluate argprefix
        if (newexp->argprefix)
            toElemDtor(newexp->argprefix);

        Logger::println("Calling constructor");
        assert(newexp->arguments != NULL);
        DtoResolveFunction(newexp->member);
        DFuncValue dfn(newexp->member, getIrFunc(newexp->member)->func, mem);
        return DtoCallFunction(newexp->loc, tc, &dfn, newexp->arguments);
    }

    assert(newexp->argprefix == NULL);

    // return default constructed class
    return new DImValue(tc, mem);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass* tc, LLValue* dst)
{
    DtoResolveClass(tc->sym);

    // Set vtable field. Doing this seperately might be optimized better.
    LLValue* tmp = DtoGEPi(dst, 0, 0, "vtbl");
    LLValue* val = DtoBitCast(getIrAggr(tc->sym)->getVtblSymbol(),
        tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);

    // For D classes, set the monitor field to null.
    const bool isCPPclass = tc->sym->isCPPclass() ? true : false;
    if (!isCPPclass)
    {
        tmp = DtoGEPi(dst, 0, 1, "monitor");
        val = LLConstant::getNullValue(tmp->getType()->getContainedType(0));
        DtoStore(val, tmp);
    }

    // Copy the rest from the static initializer, if any.
    unsigned const firstDataIdx = isCPPclass ? 1 : 2;
    uint64_t const dataBytes = tc->sym->structsize - Target::ptrsize * firstDataIdx;
    if (dataBytes == 0)
        return;

    LLValue* dstarr = DtoGEPi(dst, 0, firstDataIdx);

    // init symbols might not have valid types
    LLValue* initsym = getIrAggr(tc->sym)->getInitSymbol();
    initsym = DtoBitCast(initsym, DtoType(tc));
    LLValue* srcarr = DtoGEPi(initsym, 0, firstDataIdx);

    DtoMemCpy(dstarr, srcarr, DtoConstSize_t(dataBytes));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(Loc& loc, LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_callfinalizer");

    gIR->CreateCallOrInvoke(fn,
        DtoBitCast(inst, fn->getFunctionType()->getParamType(0)), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastClass(Loc& loc, DValue* val, Type* _to)
{
    IF_LOG Logger::println("DtoCastClass(%s, %s)", val->getType()->toChars(), _to->toChars());
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
        return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero));
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
        return DtoCastInt(loc, &im, _to);
    }
    // class -> typeof(null)
    else if (to->ty == Tnull) {
        IF_LOG Logger::println("to %s", to->toChars());
        return new DImValue(_to, LLConstant::getNullValue(DtoType(_to)));
    }

    // must be class/interface
    assert(to->ty == Tclass);
    TypeClass* tc = static_cast<TypeClass*>(to);

    // from type
    Type* from = val->getType()->toBasetype();
    TypeClass* fc = static_cast<TypeClass*>(from);

    if (fc->sym->isCPPclass()) {
        IF_LOG Logger::println("C++ class/interface, just bitcasting");
        LLValue* rval = DtoBitCast(val->getRVal(), DtoType(_to));
        return new DImValue(_to, rval);
    }

    // x -> interface
    if (InterfaceDeclaration* it = tc->sym->isInterfaceDeclaration()) {
        Logger::println("to interface");
        // interface -> interface
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("from interface");
            return DtoDynamicCastInterface(loc, val, _to);
        }
        // class -> interface - static cast
        else if (it->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast");

            // get the from class
            ClassDeclaration* cd = fc->sym->isClassDeclaration();
            DtoResolveClass(cd); // add this
            IrTypeClass* typeclass = stripModifiers(fc)->ctype->isClass();

            // find interface impl

            size_t i_index = typeclass->getInterfaceIndex(it);
            assert(i_index != ~0UL && "requesting interface that is not implemented by this class");

            // offset pointer
            LLValue* v = val->getRVal();
            LLValue* orig = v;
            v = DtoGEPi(v, 0, i_index);
            LLType* ifType = DtoType(_to);
            IF_LOG {
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
            return DtoDynamicCastObject(loc, val, _to);
        }
    }
    // x -> class
    else {
        Logger::println("to class");
        // interface -> class
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("interface cast");
            return DtoDynamicCastInterface(loc, val, _to);
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
            return DtoDynamicCastObject(loc, val, _to);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastObject(Loc& loc, DValue* val, Type* _to)
{
    // call:
    // Object _d_dynamic_cast(Object o, ClassInfo c)

    DtoResolveClass(ClassDeclaration::object);
    DtoResolveClass(Type::typeinfoclass);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_dynamic_cast");
    LLFunctionType* funcTy = func->getFunctionType();

    // Object o
    LLValue* obj = val->getRVal();
    obj = DtoBitCast(obj, funcTy->getParamType(0));
    assert(funcTy->getParamType(0) == obj->getType());

    // ClassInfo c
    TypeClass* to = static_cast<TypeClass*>(_to->toBasetype());
    DtoResolveClass(to->sym);

    LLValue* cinfo = getIrAggr(to->sym)->getClassInfoSymbol();
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));
    assert(funcTy->getParamType(1) == cinfo->getType());

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke(func, obj, cinfo).getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastInterface(Loc& loc, DValue* val, Type* _to)
{
    // call:
    // Object _d_interface_cast(void* p, ClassInfo c)

    DtoResolveClass(ClassDeclaration::object);
    DtoResolveClass(Type::typeinfoclass);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_interface_cast");
    LLFunctionType* funcTy = func->getFunctionType();

    // void* p
    LLValue* ptr = val->getRVal();
    ptr = DtoBitCast(ptr, funcTy->getParamType(0));

    // ClassInfo c
    TypeClass* to = static_cast<TypeClass*>(_to->toBasetype());
    DtoResolveClass(to->sym);
    LLValue* cinfo = getIrAggr(to->sym)->getClassInfoSymbol();
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke(func, ptr, cinfo).getInstruction();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl, char* name)
{
    // sanity checks
    assert(fdecl->isVirtual());
    assert(!fdecl->isFinalFunc());
    assert(inst->getType()->toBasetype()->ty == Tclass);
    // 0 is always ClassInfo/Interface* unless it is a CPP interface
    assert(fdecl->vtblIndex > 0 || (fdecl->vtblIndex == 0 && fdecl->linkage == LINKcpp));

    // get instance
    LLValue* vthis = inst->getRVal();
    IF_LOG Logger::cout() << "vthis: " << *vthis << '\n';

    LLValue* funcval = vthis;
    // get the vtbl for objects
    funcval = DtoGEPi(funcval, 0, 0);
    // load vtbl ptr
    funcval = DtoLoad(funcval);
    // index vtbl
    std::string vtblname = name;
    vtblname.append("@vtbl");
    funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, vtblname.c_str());
    // load funcptr
    funcval = DtoAlignedLoad(funcval);

    IF_LOG Logger::cout() << "funcval: " << *funcval << '\n';

    // cast to final funcptr type
    funcval = DtoBitCast(funcval, getPtrToType(DtoFunctionType(fdecl)));

    // postpone naming until after casting to get the name in call instructions
    funcval->setName(name);

    IF_LOG Logger::cout() << "funcval casted: " << *funcval << '\n';

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
    IrAggr* iraggr = cd->ir.irAggr;

    size_t nvars = iraggr->varDecls.size();
    std::vector<LLConstant*> arrayInits(nvars);

    for (size_t i=0; i<nvars; i++)
    {
        arrayInits[i] = build_offti_entry(cd, iraggr->varDecls[i]);
    }

    LLConstant* size = DtoConstSize_t(nvars);
    LLConstant* ptr;

    if (nvars == 0)
        return LLConstant::getNullValue( arrayT );

    // array type
    LLArrayType* arrTy = llvm::ArrayType::get(arrayInits[0]->getType(), nvars);
    LLConstant* arrInit = LLConstantArray::get(arrTy, arrayInits);

    // create symbol
    llvm::GlobalVariable* gvar = getOrCreateGlobal(cd->loc, gIR->module, arrTy,
        true, llvm::GlobalValue::InternalLinkage, arrInit, ".offti");
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

    DtoResolveFunction(dtor);
    return llvm::ConstantExpr::getBitCast(getIrFunc(dtor)->func, getPtrToType(LLType::getInt8Ty(gIR->context())));
}

static ClassFlags::Type build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code:
    // toobj.c: ToObjFile::visit(ClassDeclaration*) and ToObjFile::visit(InterfaceDeclaration*)

    ClassFlags::Type flags = ClassFlags::hasOffTi | ClassFlags::hasTypeInfo;
    if (cd->isInterfaceDeclaration())
    {
        if (cd->isCOMinterface()) flags |= ClassFlags::isCOMclass;
        return flags;
    }

    if (cd->isCOMclass()) flags |= ClassFlags::isCOMclass;
    if (cd->isCPPclass()) flags |= ClassFlags::isCPPclass;
    flags |= ClassFlags::hasGetMembers;
    if (cd->ctor)
        flags |= ClassFlags::hasCtor;
    for (ClassDeclaration *pc = cd; pc; pc = pc->baseClass)
    {
        if (pc->dtor)
        {
            flags |= ClassFlags::hasDtor;
            break;
        }
    }
    if (cd->isabstract)
        flags |= ClassFlags::isAbstract;
    for (ClassDeclaration *pc = cd; pc; pc = pc->baseClass)
    {
        if (pc->members)
        {
            for (size_t i = 0; i < pc->members->dim; i++)
            {
                Dsymbol *sm = (*pc->members)[i];
                //printf("sm = %s %s\n", sm->kind(), sm->toChars());
                if (sm->hasPointers())
                    return flags;
            }
        }
    }
    flags |= ClassFlags::noPointers;

    return flags;
}

LLConstant* DtoDefineClassInfo(ClassDeclaration* cd)
{
//     The layout is:
//       {
//         void **vptr;
//         monitor_t monitor;
//         byte[] init;
//         string name;
//         void*[] vtbl;
//         Interface[] interfaces;
//         TypeInfo_Class base;
//         void *destructor;
//         void function(Object) classInvariant;
//         ClassFlags m_flags;
//         void* deallocator;
//         OffsetTypeInfo[] m_offTi;
//         void function(Object) defaultConstructor;
//         immutable(void)* m_RTInfo;
//       }

    IF_LOG Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);

    IrAggr* ir = getIrAggr(cd);
    ClassDeclaration* cinfo = Type::typeinfoclass;

    if (cinfo->fields.dim != 12)
    {
        error(Loc(), "Unexpected number of fields in object.ClassInfo; "
            "druntime version does not match compiler (see -v)");
        fatal();
    }

    // use the rtti builder
    RTTIBuilder b(cinfo);

    LLConstant* c;

    LLType* voidPtr = getVoidPtrType();
    LLType* voidPtrPtr = getPtrToType(voidPtr);

    // adapted from original dmd code
    // init[]
    if (cd->isInterfaceDeclaration())
    {
        b.push_null_void_array();
    }
    else
    {
        size_t initsz = cd->size(Loc());
        b.push_void_array(initsz, ir->getInitSymbol());
    }

    // name[]
    const char *name = cd->ident->toChars();
    size_t namelen = strlen(name);
    if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0))
    {
        name = cd->toPrettyChars();
        namelen = strlen(name);
    }
    b.push_string(name);

    // vtbl[]
    if (cd->isInterfaceDeclaration())
    {
        b.push_array(0, getNullValue(voidPtrPtr));
    }
    else
    {
        c = DtoBitCast(ir->getVtblSymbol(), voidPtrPtr);
        b.push_array(cd->vtbl.dim, c);
    }

    // interfaces[]
    b.push(ir->getClassInfoInterfaces());

    // base
    // interfaces never get a base, just the interfaces[]
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
    VarDeclaration* invVar = cinfo->fields[6];
    b.push_funcptr(cd->inv, invVar->type);

    // flags
    ClassFlags::Type flags = build_classinfo_flags(cd);
    b.push_uint(flags);

    // deallocator
    b.push_funcptr(cd->aggDelete, Type::tvoid->pointerTo());

    // offset typeinfo
    VarDeclaration* offTiVar = cinfo->fields[9];
#if GENERATE_OFFTI
    if (cd->isInterfaceDeclaration())
        b.push_null(offTiVar->type);
    else
        b.push(build_offti_array(cd, DtoType(offTiVar->type)));
#else
    b.push_null(offTiVar->type);
#endif

    // defaultConstructor
    VarDeclaration* defConstructorVar = cinfo->fields.data[10];
    CtorDeclaration* defConstructor = cd->defaultCtor;
    if (defConstructor && (defConstructor->storage_class & STCdisable))
        defConstructor = NULL;
    b.push_funcptr(defConstructor, defConstructorVar->type);

    // m_RTInfo
    // The cases where getRTInfo is null are not quite here, but the code is
    // modelled after what DMD does.
    if (cd->getRTInfo)
        b.push(toConstElem(cd->getRTInfo, gIR));
    else if (flags & ClassFlags::noPointers)
        b.push_size_as_vp(0);       // no pointers
    else
        b.push_size_as_vp(1);       // has pointers

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
