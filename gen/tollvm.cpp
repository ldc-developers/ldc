
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
#include "gen/linkage.h"
#include "gen/llvm-version.h"

#include "ir/irtype.h"
#include "ir/irtypeclass.h"
#include "ir/irtypefunction.h"

bool DtoIsPassedByRef(Type* type)
{
    Type* typ = type->toBasetype();
    TY t = typ->ty;
    return (t == Tstruct || t == Tsarray);
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
            return llvm::Attribute::SExt;

        case Tuns8:
        case Tuns16:
            return llvm::Attribute::ZExt;
        }
    }
    return llvm::Attribute::None;
}

const LLType* DtoType(Type* t)
{
#if DMDV2
    t = t->mutableOf();
#endif

    if (t->irtype)
    {
        return t->irtype->get();
    }

    IF_LOG Logger::println("Building type: %s", t->toChars());

    assert(t);
    switch (t->ty)
    {
    // basic types
    case Tvoid:
    case Tint8:
    case Tuns8:
    case Tint16:
    case Tuns16:
    case Tint32:
    case Tuns32:
    case Tint64:
    case Tuns64:
    case Tfloat32:
    case Tfloat64:
    case Tfloat80:
    case Timaginary32:
    case Timaginary64:
    case Timaginary80:
    case Tcomplex32:
    case Tcomplex64:
    case Tcomplex80:
    //case Tbit:
    case Tbool:
    case Tchar:
    case Twchar:
    case Tdchar:
    {
        t->irtype = new IrTypeBasic(t);
        return t->irtype->buildType();
    }

    // pointers
    case Tpointer:
    {
        t->irtype = new IrTypePointer(t);
        return t->irtype->buildType();
    }

    // arrays
    case Tarray:
    {
        t->irtype = new IrTypeArray(t);
        return t->irtype->buildType();
    }

    case Tsarray:
    {
        t->irtype = new IrTypeSArray(t);
        return t->irtype->buildType();
    }

    // aggregates
    case Tstruct:    {
        TypeStruct* ts = (TypeStruct*)t;
        t->irtype = new IrTypeStruct(ts->sym);
        return t->irtype->buildType();
    }
    case Tclass:    {
        TypeClass* tc = (TypeClass*)t;
        t->irtype = new IrTypeClass(tc->sym);
        return t->irtype->buildType();
    }

    // functions
    case Tfunction:
    {
        t->irtype = new IrTypeFunction(t);
        return t->irtype->buildType();
    }

    // delegates
    case Tdelegate:
    {
        t->irtype = new IrTypeDelegate(t);
        return t->irtype->buildType();
    }

    // typedefs
    // enum

    // FIXME: maybe just call toBasetype first ?
    case Ttypedef:
    case Tenum:
    {
        Type* bt = t->toBasetype();
        assert(bt);
        return DtoType(bt);
    }

    // associative arrays
    case Taarray:
        return getVoidPtrType();

/*
    Not needed atm as VarDecls for tuples are rewritten as a string of 
    VarDecls for the fields (u -> _u_field_0, ...)

    case Ttuple:
    {
        TypeTuple* ttupl = (TypeTuple*)t;
        return DtoStructTypeFromArguments(ttupl->arguments);
    }
*/

    default:
        printf("trying to convert unknown type '%s' with value %d\n", t->toChars(), t->ty);
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
const LLType* DtoStructTypeFromArguments(Arguments* arguments)
{
    if (!arguments)
        return LLType::VoidTy;

    std::vector<const LLType*> types;
    for (size_t i = 0; i < arguments->dim; i++)
    {
        Argument *arg = (Argument *)arguments->data[i];
        assert(arg && arg->type);

        types.push_back(DtoType(arg->type));
    }
    return LLStructType::get(types);
}
*/

//////////////////////////////////////////////////////////////////////////////////////////

const LLType* DtoTypeNotVoid(Type* t)
{
    const LLType* lt = DtoType(t);
    if (lt == LLType::VoidTy)
        return LLType::Int8Ty;
    return lt;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoDelegateEquals(TOK op, LLValue* lhs, LLValue* rhs)
{
    Logger::println("Doing delegate equality");
    llvm::Value *b1, *b2;
    if (rhs == NULL)
    {
        rhs = LLConstant::getNullValue(lhs->getType());
    }

    LLValue* l = gIR->ir->CreateExtractValue(lhs, 0);
    LLValue* r = gIR->ir->CreateExtractValue(rhs, 0);
    b1 = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,l,r,"tmp");

    l = gIR->ir->CreateExtractValue(lhs, 1);
    r = gIR->ir->CreateExtractValue(rhs, 1);
    b2 = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,l,r,"tmp");

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
        if (needsTemplateLinkage(sym))
            return TEMPLATE_LINKAGE_TYPE;
    }
    // function
    else if (FuncDeclaration* fdecl = sym->isFuncDeclaration())
    {
        assert(fdecl->type->ty == Tfunction);
        TypeFunction* ft = (TypeFunction*)fdecl->type;

        // array operations are always template linkage
        if (fdecl->isArrayOp)
            return TEMPLATE_LINKAGE_TYPE;
        // intrinsics are always external
        if (fdecl->llvmInternal == LLVMintrinsic)
            return llvm::GlobalValue::ExternalLinkage;
        // template instances should have weak linkage
        // but only if there's a body, and it's not naked
        // otherwise we make it external
        else if (needsTemplateLinkage(fdecl) && fdecl->fbody && !fdecl->naked)
            return TEMPLATE_LINKAGE_TYPE;
        // extern(C) functions are always external
        else if (ft->linkage == LINKc)
            return llvm::GlobalValue::ExternalLinkage;
        // Function & delegate literals, foreach bodies and lazy parameters: internal linkage
        else if (fdecl->isFuncLiteralDeclaration())
            return llvm::GlobalValue::InternalLinkage;
    }
    // class
    else if (ClassDeclaration* cd = sym->isClassDeclaration())
    {
        // template
        if (needsTemplateLinkage(cd))
            return TEMPLATE_LINKAGE_TYPE;
    }
    else
    {
        assert(0 && "not global/function");
    }
    
    // The following breaks for nested naked functions, so check for that.
    bool skipNestedCheck = false;
    if (FuncDeclaration* fd = sym->isFuncDeclaration())
        skipNestedCheck = (fd->naked != 0);
    
    // Any symbol nested in a function can't be referenced directly from
    // outside that function, so we can give such symbols internal linkage.
    // This holds even if nested indirectly, such as member functions of
    // aggregates nested in functions.
    //
    // Note: This must be checked after things like template member-ness or
    // symbols nested in templates would get duplicated for each module,
    // breaking things like
    // ---
    // int counter(T)() { static int i; return i++; }"
    // ---
    // if instances get emitted in multiple object files because they'd use
    // different instances of 'i'.
    if (!skipNestedCheck)
        for (Dsymbol* parent = sym->parent; parent ; parent = parent->parent) {
            if (parent->isFuncDeclaration())
                return llvm::GlobalValue::InternalLinkage;
        }
    
    // default to external linkage
    return llvm::GlobalValue::ExternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym)
{
    if (needsTemplateLinkage(sym))
        return TEMPLATE_LINKAGE_TYPE;
    else
        return llvm::GlobalValue::InternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym)
{
    if (needsTemplateLinkage(sym))
        return TEMPLATE_LINKAGE_TYPE;
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
        if (Logger::enabled())
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

    const LLType* intTy = DtoSize_t();
    llvm::Function* fn = llvm::Intrinsic::getDeclaration(gIR->module,
        llvm::Intrinsic::memset, &intTy, 1);

    gIR->ir->CreateCall4(fn, dst, DtoConstUbyte(0), nbytes, DtoConstUint(0), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes, unsigned align)
{
    dst = DtoBitCast(dst,getVoidPtrType());
    src = DtoBitCast(src,getVoidPtrType());

    const LLType* intTy = DtoSize_t();
    llvm::Function* fn = llvm::Intrinsic::getDeclaration(gIR->module,
        llvm::Intrinsic::memcpy, &intTy, 1);

    gIR->ir->CreateCall4(fn, dst, src, nbytes, DtoConstUint(align), "");
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

    if(llty == LLType::FloatTy || llty == LLType::DoubleTy)
        return LLConstantFP::get(llty, value);
    else if(llty == LLType::X86_FP80Ty) {
        uint64_t bits[] = {0, 0};
        bits[0] = *(uint64_t*)&value;
        bits[1] = *(uint16_t*)((uint64_t*)&value + 1);
        return LLConstantFP::get(APFloat(APInt(80, 2, bits)));
    } else {
        assert(0 && "Unknown floating point type encountered");
    }
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
//     if (Logger::enabled())
//         Logger::cout() << "loading " << *src <<  '\n';
    llvm::LoadInst* ld = gIR->ir->CreateLoad(src, name ? name : "tmp");
    //ld->setVolatile(gIR->func()->inVolatile);
    return ld;
}

// Like DtoLoad, but the pointer is guaranteed to be aligned appropriately for the type.
LLValue* DtoAlignedLoad(LLValue* src, const char* name)
{
    llvm::LoadInst* ld = gIR->ir->CreateLoad(src, name ? name : "tmp");
    ld->setAlignment(getABITypeAlign(ld->getType()));
    return ld;
}


void DtoStore(LLValue* src, LLValue* dst)
{
//     if (Logger::enabled())
//         Logger::cout() << "storing " << *src << " into " << *dst << '\n';
    LLValue* st = gIR->ir->CreateStore(src,dst);
    //st->setVolatile(gIR->func()->inVolatile);
}

// Like DtoStore, but the pointer is guaranteed to be aligned appropriately for the type.
void DtoAlignedStore(LLValue* src, LLValue* dst)
{
    llvm::StoreInst* st = gIR->ir->CreateStore(src,dst);
    st->setAlignment(getABITypeAlign(src->getType()));
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoBitCast(LLValue* v, const LLType* t, const char* name)
{
    if (v->getType() == t)
        return v;
    assert(!isaStruct(t));
    return gIR->ir->CreateBitCast(v, t, name ? name : "tmp");
}

LLConstant* DtoBitCast(LLConstant* v, const LLType* t)
{
    if (v->getType() == t)
        return v;
    return llvm::ConstantExpr::getBitCast(v, t);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoInsertValue(LLValue* aggr, LLValue* v, unsigned idx)
{
    return gIR->ir->CreateInsertValue(aggr, v, idx);
}

LLValue* DtoExtractValue(LLValue* aggr, unsigned idx)
{
    return gIR->ir->CreateExtractValue(aggr, idx);
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

LLConstant* getNullValue(const LLType* t)
{
    return LLConstant::getNullValue(t);
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

size_t getTypePaddedSize(const LLType* t)
{
#if LLVM_REV > 71348
    size_t sz = gTargetData->getTypeAllocSize(t);
#else
    size_t sz = gTargetData->getTypePaddedSize(t);
#endif
    //Logger::cout() << "abi type size of: " << *t << " == " << sz << '\n';
    return sz;
}

unsigned char getABITypeAlign(const LLType* t)
{
    return gTargetData->getABITypeAlignment(t);
}

unsigned char getPrefTypeAlign(const LLType* t)
{
    return gTargetData->getPrefTypeAlignment(t);
}

const LLType* getBiggestType(const LLType** begin, size_t n)
{
    const LLType* bigTy = 0;
    size_t bigSize = 0;
    size_t bigAlign = 0;

    const LLType** end = begin+n;
    while (begin != end)
    {
        const LLType* T = *begin;

        size_t sz = getTypePaddedSize(T);
        size_t ali = getABITypeAlign(T);
        if (sz > bigSize || (sz == bigSize && ali > bigAlign))
        {
            bigTy = T;
            bigSize = sz;
            bigAlign = ali;
        }

        ++begin;
    }

    // will be null for n==0
    return bigTy;
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
    types.push_back(DtoType(cd2->type));
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
    if (global.params.os == OSWindows)
    {
        // CRITICAL_SECTION.sizeof == 68
        std::vector<const LLType*> types(17, LLType::Int32Ty);
        return LLStructType::get(types);
    }

    // FreeBSD
    else if (global.params.os == OSFreeBSD) {
        // Just a pointer
        return LLStructType::get(DtoSize_t(), NULL);
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

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoAggrPair(const LLType* type, LLValue* V1, LLValue* V2, const char* name)
{
    LLValue* res = llvm::UndefValue::get(type);
    res = gIR->ir->CreateInsertValue(res, V1, 0, "tmp");
    return gIR->ir->CreateInsertValue(res, V2, 1, name?name:"tmp");
}

LLValue* DtoAggrPair(LLValue* V1, LLValue* V2, const char* name)
{
    const LLType* t = LLStructType::get(V1->getType(), V2->getType(), NULL);
    return DtoAggrPair(t, V1, V2, name);
}

LLValue* DtoAggrPaint(LLValue* aggr, const LLType* as)
{
    if (aggr->getType() == as)
        return aggr;

    LLValue* res = llvm::UndefValue::get(as);

    LLValue* V = gIR->ir->CreateExtractValue(aggr, 0, "tmp");;
    V = DtoBitCast(V, as->getContainedType(0));
    res = gIR->ir->CreateInsertValue(res, V, 0, "tmp");

    V = gIR->ir->CreateExtractValue(aggr, 1, "tmp");;
    V = DtoBitCast(V, as->getContainedType(1));
    return gIR->ir->CreateInsertValue(res, V, 1, "tmp");
}

LLValue* DtoAggrPairSwap(LLValue* aggr)
{
    Logger::println("swapping aggr pair");
    LLValue* r = gIR->ir->CreateExtractValue(aggr, 0);
    LLValue* i = gIR->ir->CreateExtractValue(aggr, 1);
    return DtoAggrPair(i, r, "swapped");
}
