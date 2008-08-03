#include <iostream>

#include "gen/llvm.h"

#include "dsymbol.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"
#include "module.h"

#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/typeinf.h"
#include "gen/complex.h"
#include "gen/llvmhelpers.h"

bool DtoIsPassedByRef(Type* type)
{
    Type* typ = DtoDType(type);
    TY t = typ->ty;
    return (t == Tstruct || t == Tarray || t == Tdelegate || t == Tsarray || typ->iscomplex());
}

bool DtoIsReturnedInArg(Type* type)
{
    Type* typ = DtoDType(type);
    TY t = typ->ty;
    return (t == Tstruct || t == Tarray || t == Tdelegate || t == Tsarray || typ->iscomplex());
}

unsigned DtoShouldExtend(Type* type)
{
    type = type->toBasetype();
    if (type->isintegral())
    {
        switch(type->ty)
        {
        case Tint8:
        case Tint16:
            return llvm::ParamAttr::SExt;

        case Tuns8:
        case Tuns16:
            return llvm::ParamAttr::ZExt;
        }
    }
    return llvm::ParamAttr::None;
}

Type* DtoDType(Type* t)
{
    if (t->ty == Ttypedef) {
        Type* bt = t->toBasetype();
        assert(bt);
        return DtoDType(bt);
    }
    return t;
}

const LLType* DtoType(Type* t)
{
    assert(t);
    switch (t->ty)
    {
    // integers
    case Tint8:
    case Tuns8:
    case Tchar:
        return (const LLType*)LLType::Int8Ty;
    case Tint16:
    case Tuns16:
    case Twchar:
        return (const LLType*)LLType::Int16Ty;
    case Tint32:
    case Tuns32:
    case Tdchar:
        return (const LLType*)LLType::Int32Ty;
    case Tint64:
    case Tuns64:
        return (const LLType*)LLType::Int64Ty;

    case Tbool:
        return (const LLType*)llvm::ConstantInt::getTrue()->getType();

    // floats
    case Tfloat32:
    case Timaginary32:
        return LLType::FloatTy;
    case Tfloat64:
    case Timaginary64:
        return LLType::DoubleTy;
    case Tfloat80:
    case Timaginary80:
        if (global.params.cpu == ARCHx86)
            return LLType::X86_FP80Ty;
        else
            return LLType::DoubleTy;

    // complex
    case Tcomplex32:
    case Tcomplex64:
    case Tcomplex80:
        return DtoComplexType(t);

    // pointers
    case Tpointer:
        // getPtrToType checks for void itself
        return getPtrToType(DtoType(t->next));

    // arrays
    case Tarray:
        return DtoArrayType(t);
    case Tsarray:
        return DtoStaticArrayType(t);

    // void
    case Tvoid:
        return LLType::VoidTy;

    // aggregates
    case Tstruct:    {
        TypeStruct* ts = (TypeStruct*)t;
        assert(ts->sym);
        DtoResolveDsymbol(ts->sym);
        return ts->sym->ir.irStruct->recty.get(); // t->ir.type->get();
    }

    case Tclass:    {
        TypeClass* tc = (TypeClass*)t;
        assert(tc->sym);
        DtoResolveDsymbol(tc->sym);
        return getPtrToType(tc->sym->ir.irStruct->recty.get()); // t->ir.type->get());
    }

    // functions
    case Tfunction:
    {
        if (!t->ir.type || *t->ir.type == NULL) {
            return DtoFunctionType(t,NULL);
        }
        else {
            return t->ir.type->get();
        }
    }

    // delegates
    case Tdelegate:
    {
        if (!t->ir.type || *t->ir.type == NULL) {
            return DtoDelegateType(t);
        }
        else {
            return t->ir.type->get();
        }
    }

    // typedefs
    // enum
    case Ttypedef:
    case Tenum:
    {
        Type* bt = t->toBasetype();
        assert(bt);
        return DtoType(bt);
    }

    // associative arrays
    case Taarray:
    {
        TypeAArray* taa = (TypeAArray*)t;
        // aa key/val can't be void
        return getPtrToType(LLStructType::get(DtoType(taa->key), DtoType(taa->next), 0));
    }

    // opaque type
    case Topaque:
        return llvm::OpaqueType::get();

    default:
        printf("trying to convert unknown type with value %d\n", t->ty);
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLType* DtoTypeNotVoid(Type* t)
{
    const LLType* lt = DtoType(t);
    if (lt == LLType::VoidTy)
        return LLType::Int8Ty;
    return lt;
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLStructType* DtoDelegateType(Type* t)
{
    const LLType* i8ptr = getVoidPtrType();
    const LLType* func = DtoFunctionType(t->next, i8ptr);
    const LLType* funcptr = getPtrToType(func);
    return LLStructType::get(i8ptr, funcptr, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoDelegateEquals(TOK op, LLValue* lhs, LLValue* rhs)
{
    Logger::println("Doing delegate equality");
    llvm::ICmpInst::Predicate pred = (op == TOKequal || op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    llvm::Value *b1, *b2;
    if (rhs == NULL)
    {
        LLValue* l = DtoLoad(DtoGEPi(lhs,0,0));
        LLValue* r = llvm::Constant::getNullValue(l->getType());
        b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
        l = DtoLoad(DtoGEPi(lhs,0,1));
        r = llvm::Constant::getNullValue(l->getType());
        b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    }
    else
    {
        LLValue* l = DtoLoad(DtoGEPi(lhs,0,0));
        LLValue* r = DtoLoad(DtoGEPi(rhs,0,0));
        b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
        l = DtoLoad(DtoGEPi(lhs,0,1));
        r = DtoLoad(DtoGEPi(rhs,0,1));
        b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    }
    LLValue* b = gIR->ir->CreateAnd(b1,b2,"tmp");
    if (op == TOKnotequal || op == TOKnotidentity)
        return gIR->ir->CreateNot(b,"tmp");
    return b;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLGlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym)
{
    // global variable
    if (VarDeclaration* vd = sym->isVarDeclaration())
    {
        // template
        if (DtoIsTemplateInstance(sym))
            return llvm::GlobalValue::WeakLinkage;
        // local static
        else if (sym->parent && sym->parent->isFuncDeclaration())
            return llvm::GlobalValue::InternalLinkage;
    }
    // function
    else if (FuncDeclaration* fdecl = sym->isFuncDeclaration())
    {
        assert(fdecl->type->ty == Tfunction);
        TypeFunction* ft = (TypeFunction*)fdecl->type;

        // intrinsics are always external
        if (fdecl->llvmInternal == LLVMintrinsic)
            return llvm::GlobalValue::ExternalLinkage;
        // template instances should have weak linkage
        else if (DtoIsTemplateInstance(fdecl))
            return llvm::GlobalValue::WeakLinkage;
        // extern(C) functions are always external
        else if (ft->linkage == LINKc)
            return llvm::GlobalValue::ExternalLinkage;
    }
    // class
    else if (ClassDeclaration* cd = sym->isClassDeclaration())
    {
        // template
        if (DtoIsTemplateInstance(cd))
            return llvm::GlobalValue::WeakLinkage;
    }
    else
    {
        assert(0 && "not global/function");
    }

    // default to external linkage
    return llvm::GlobalValue::ExternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym)
{
    if (DtoIsTemplateInstance(sym))
        return llvm::GlobalValue::WeakLinkage;
    else
        return llvm::GlobalValue::InternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym)
{
    if (DtoIsTemplateInstance(sym))
        return llvm::GlobalValue::WeakLinkage;
    else
        return llvm::GlobalValue::ExternalLinkage;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoPointedType(LLValue* ptr, LLValue* val)
{
    const LLType* ptrTy = ptr->getType()->getContainedType(0);
    const LLType* valTy = val->getType();
    // ptr points to val's type
    if (ptrTy == valTy)
    {
        return val;
    }
    // ptr is integer pointer
    else if (ptrTy->isInteger())
    {
        // val is integer
        assert(valTy->isInteger());
        const LLIntegerType* pt = llvm::cast<const LLIntegerType>(ptrTy);
        const LLIntegerType* vt = llvm::cast<const LLIntegerType>(valTy);
        if (pt->getBitWidth() < vt->getBitWidth()) {
            return new llvm::TruncInst(val, pt, "tmp", gIR->scopebb());
        }
        else
        assert(0);
    }
    // something else unsupported
    else
    {
        Logger::cout() << *ptrTy << '|' << *valTy << '\n';
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLType* DtoSize_t()
{
    // the type of size_t does not change once set
    static const LLType* t = NULL;
    if (t == NULL)
        t = (global.params.is64bit) ? LLType::Int64Ty : LLType::Int32Ty;
    return t;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP1(LLValue* ptr, LLValue* i0, const char* var, llvm::BasicBlock* bb)
{
    return llvm::GetElementPtrInst::Create(ptr, i0, var?var:"tmp", bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var, llvm::BasicBlock* bb)
{
    LLSmallVector<LLValue*,2> v(2);
    v[0] = i0;
    v[1] = i1;
    return llvm::GetElementPtrInst::Create(ptr, v.begin(), v.end(), var?var:"tmp", bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, const DStructIndexVector& src, const char* var, llvm::BasicBlock* bb)
{
    size_t n = src.size();
    LLSmallVector<LLValue*, 3> dst(n);

    size_t j=0;
    for (DStructIndexVector::const_iterator i=src.begin(); i!=src.end(); ++i)
        dst[j++] = DtoConstUint(*i);

    return llvm::GetElementPtrInst::Create(ptr, dst.begin(), dst.end(), var?var:"tmp", bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi1(LLValue* ptr, unsigned i, const char* var, llvm::BasicBlock* bb)
{
    return llvm::GetElementPtrInst::Create(ptr, DtoConstUint(i), var?var:"tmp", bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var, llvm::BasicBlock* bb)
{
    LLSmallVector<LLValue*,2> v(2);
    v[0] = DtoConstUint(i0);
    v[1] = DtoConstUint(i1);
    return llvm::GetElementPtrInst::Create(ptr, v.begin(), v.end(), var?var:"tmp", bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(LLValue* dst, LLValue* nbytes)
{
    dst = DtoBitCast(dst,getVoidPtrType());

    llvm::Function* fn;
    if (global.params.is64bit)
        fn = GET_INTRINSIC_DECL(memset_i64);
    else
        fn = GET_INTRINSIC_DECL(memset_i32);

    gIR->ir->CreateCall4(fn, dst, DtoConstUbyte(0), nbytes, DtoConstUint(0), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes)
{
    dst = DtoBitCast(dst,getVoidPtrType());
    src = DtoBitCast(src,getVoidPtrType());

    llvm::Function* fn;
    if (global.params.is64bit)
        fn = GET_INTRINSIC_DECL(memcpy_i64);
    else
        fn = GET_INTRINSIC_DECL(memcpy_i32);

    gIR->ir->CreateCall4(fn, dst, src, nbytes, DtoConstUint(0), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoMemCmp(LLValue* lhs, LLValue* rhs, LLValue* nbytes)
{
    // int memcmp ( const void * ptr1, const void * ptr2, size_t num );

    LLFunction* fn = gIR->module->getFunction("memcmp");
    if (!fn)
    {
        std::vector<const LLType*> params(3);
        params[0] = getVoidPtrType();
        params[1] = getVoidPtrType();
        params[2] = DtoSize_t();
        const LLFunctionType* fty = LLFunctionType::get(LLType::Int32Ty, params, false);
        fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, "memcmp", gIR->module);
    }

    lhs = DtoBitCast(lhs,getVoidPtrType());
    rhs = DtoBitCast(rhs,getVoidPtrType());

    return gIR->ir->CreateCall3(fn, lhs, rhs, nbytes, "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAggrZeroInit(LLValue* v)
{
    uint64_t n = getTypeStoreSize(v->getType()->getContainedType(0));
    DtoMemSetZero(v, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAggrCopy(LLValue* dst, LLValue* src)
{
    uint64_t n = getTypeStoreSize(dst->getType()->getContainedType(0));
    DtoMemCpy(dst, src, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemoryBarrier(bool ll, bool ls, bool sl, bool ss, bool device)
{
    llvm::Function* fn = GET_INTRINSIC_DECL(memory_barrier);
    assert(fn != NULL);

    LLSmallVector<LLValue*, 5> llargs;
    llargs.push_back(DtoConstBool(ll));
    llargs.push_back(DtoConstBool(ls));
    llargs.push_back(DtoConstBool(sl));
    llargs.push_back(DtoConstBool(ss));
    llargs.push_back(DtoConstBool(device));

    llvm::CallInst::Create(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::ConstantInt* DtoConstSize_t(size_t i)
{
    return llvm::ConstantInt::get(DtoSize_t(), i, false);
}
llvm::ConstantInt* DtoConstUint(unsigned i)
{
    return llvm::ConstantInt::get(LLType::Int32Ty, i, false);
}
llvm::ConstantInt* DtoConstInt(int i)
{
    return llvm::ConstantInt::get(LLType::Int32Ty, i, true);
}
LLConstant* DtoConstBool(bool b)
{
    return llvm::ConstantInt::get(LLType::Int1Ty, b, false);
}
llvm::ConstantInt* DtoConstUbyte(unsigned char i)
{
    return llvm::ConstantInt::get(LLType::Int8Ty, i, false);
}

llvm::ConstantFP* DtoConstFP(Type* t, long double value)
{
    const LLType* llty = DtoType(t);
    assert(llty->isFloatingPoint());
    return LLConstantFP::get(llty, value);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstString(const char* str)
{
    std::string s(str?str:"");
    LLConstant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, ".str", gIR->module);
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    return DtoConstSlice(
        DtoConstSize_t(s.length()),
        llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2)
    );
}
LLConstant* DtoConstStringPtr(const char* str, const char* section)
{
    std::string s(str);
    LLConstant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, ".str", gIR->module);
    if (section) gvar->setSection(section);
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    return llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoLoad(LLValue* src, const char* name)
{
    LLValue* ld = gIR->ir->CreateLoad(src, name ? name : "tmp");
    //ld->setVolatile(gIR->func()->inVolatile);
    return ld;
}

void DtoStore(LLValue* src, LLValue* dst)
{
    LLValue* st = gIR->ir->CreateStore(src,dst);
    //st->setVolatile(gIR->func()->inVolatile);
}

bool DtoCanLoad(LLValue* ptr)
{
    if (isaPointer(ptr->getType())) {
        const LLType* data = ptr->getType()->getContainedType(0);
        return data->isFirstClassType() && !(isaStruct(data) || isaArray(data));
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoBitCast(LLValue* v, const LLType* t, const char* name)
{
    if (v->getType() == t)
        return v;
    return gIR->ir->CreateBitCast(v, t, name ? name : "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLPointerType* isaPointer(LLValue* v)
{
    return llvm::dyn_cast<LLPointerType>(v->getType());
}

const LLPointerType* isaPointer(const LLType* t)
{
    return llvm::dyn_cast<LLPointerType>(t);
}

const LLArrayType* isaArray(LLValue* v)
{
    return llvm::dyn_cast<LLArrayType>(v->getType());
}

const LLArrayType* isaArray(const LLType* t)
{
    return llvm::dyn_cast<LLArrayType>(t);
}

const LLStructType* isaStruct(LLValue* v)
{
    return llvm::dyn_cast<LLStructType>(v->getType());
}

const LLStructType* isaStruct(const LLType* t)
{
    return llvm::dyn_cast<LLStructType>(t);
}

const LLFunctionType* isaFunction(LLValue* v)
{
    return llvm::dyn_cast<LLFunctionType>(v->getType());
}

const LLFunctionType* isaFunction(const LLType* t)
{
    return llvm::dyn_cast<LLFunctionType>(t);
}

LLConstant* isaConstant(LLValue* v)
{
    return llvm::dyn_cast<llvm::Constant>(v);
}

llvm::ConstantInt* isaConstantInt(LLValue* v)
{
    return llvm::dyn_cast<llvm::ConstantInt>(v);
}

llvm::Argument* isaArgument(LLValue* v)
{
    return llvm::dyn_cast<llvm::Argument>(v);
}

llvm::GlobalVariable* isaGlobalVar(LLValue* v)
{
    return llvm::dyn_cast<llvm::GlobalVariable>(v);
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLPointerType* getPtrToType(const LLType* t)
{
    if (t == LLType::VoidTy)
        t = LLType::Int8Ty;
    return LLPointerType::get(t, 0);
}

const LLPointerType* getVoidPtrType()
{
    return getPtrToType(LLType::Int8Ty);
}

llvm::ConstantPointerNull* getNullPtr(const LLType* t)
{
    const LLPointerType* pt = llvm::cast<LLPointerType>(t);
    return llvm::ConstantPointerNull::get(pt);
}

//////////////////////////////////////////////////////////////////////////////////////////

size_t getTypeBitSize(const LLType* t)
{
    return gTargetData->getTypeSizeInBits(t);
}

size_t getTypeStoreSize(const LLType* t)
{
    return gTargetData->getTypeStoreSize(t);
}

size_t getABITypeSize(const LLType* t)
{
    Logger::cout() << "getting abi type of: " << *t << '\n';
    return gTargetData->getABITypeSize(t);
}

unsigned char getABITypeAlign(const LLType* t)
{
    return gTargetData->getABITypeAlignment(t);
}

unsigned char getPrefTypeAlign(const LLType* t)
{
    return gTargetData->getPrefTypeAlignment(t);
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLStructType* DtoInterfaceInfoType()
{
    if (gIR->interfaceInfoType)
        return gIR->interfaceInfoType;

    // build interface info type
    std::vector<const LLType*> types;
    // ClassInfo classinfo
    ClassDeclaration* cd2 = ClassDeclaration::classinfo;
    DtoResolveClass(cd2);
    types.push_back(getPtrToType(cd2->type->ir.type->get()));
    // void*[] vtbl
    std::vector<const LLType*> vtbltypes;
    vtbltypes.push_back(DtoSize_t());
    const LLType* byteptrptrty = getPtrToType(getPtrToType(LLType::Int8Ty));
    vtbltypes.push_back(byteptrptrty);
    types.push_back(LLStructType::get(vtbltypes));
    // int offset
    types.push_back(LLType::Int32Ty);
    // create type
    gIR->interfaceInfoType = LLStructType::get(types);

    return gIR->interfaceInfoType;
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLStructType* DtoMutexType()
{
    if (gIR->mutexType)
        return gIR->mutexType;

    // win32
    if (global.params.isWindows)
    {
        // CRITICAL_SECTION.sizeof == 68
        std::vector<const LLType*> types(17, LLType::Int32Ty);
        return LLStructType::get(types);
    }

    // pthread_fastlock
    std::vector<const LLType*> types2;
    types2.push_back(DtoSize_t());
    types2.push_back(LLType::Int32Ty);
    const LLStructType* fastlock = LLStructType::get(types2);

    // pthread_mutex
    std::vector<const LLType*> types1;
    types1.push_back(LLType::Int32Ty);
    types1.push_back(LLType::Int32Ty);
    types1.push_back(getVoidPtrType());
    types1.push_back(LLType::Int32Ty);
    types1.push_back(fastlock);
    const LLStructType* pmutex = LLStructType::get(types1);

    // D_CRITICAL_SECTION
    LLOpaqueType* opaque = LLOpaqueType::get();
    std::vector<const LLType*> types;
    types.push_back(getPtrToType(opaque));
    types.push_back(pmutex);

    // resolve type
    pmutex = LLStructType::get(types);
    LLPATypeHolder pa(pmutex);
    opaque->refineAbstractTypeTo(pa.get());
    pmutex = isaStruct(pa.get());

    gIR->mutexType = pmutex;
    gIR->module->addTypeName("D_CRITICAL_SECTION", pmutex);
    return pmutex;
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLStructType* DtoModuleReferenceType()
{
    if (gIR->moduleRefType)
        return gIR->moduleRefType;

    // this is a recursive type so start out with the opaque
    LLOpaqueType* opaque = LLOpaqueType::get();

    // add members
    std::vector<const LLType*> types;
    types.push_back(getPtrToType(opaque));
    types.push_back(DtoType(Module::moduleinfo->type));

    // resolve type
    const LLStructType* st = LLStructType::get(types);
    LLPATypeHolder pa(st);
    opaque->refineAbstractTypeTo(pa.get());
    st = isaStruct(pa.get());

    // done
    gIR->moduleRefType = st;
    gIR->module->addTypeName("ModuleReference", st);
    return st;
}
