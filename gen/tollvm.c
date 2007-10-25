#include <iostream>

#include "gen/llvm.h"

#include "mtype.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"

#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/elem.h"
#include "gen/arrays.h"

bool LLVM_DtoIsPassedByRef(Type* type)
{
    TY t = type->ty;
    if (t == Tstruct || t == Tarray || t == Tdelegate)
        return true;
    else if (t == Ttypedef) {
        Type* bt = type->toBasetype();
        assert(bt);
        return LLVM_DtoIsPassedByRef(bt);
    }
    return false;
}

Type* LLVM_DtoDType(Type* t)
{
    if (t->ty == Ttypedef) {
        Type* bt = t->toBasetype();
        assert(bt);
        return LLVM_DtoDType(bt);
    }
    return t;
}

const llvm::Type* LLVM_DtoType(Type* t)
{
    assert(t);
    switch (t->ty)
    {
    // integers
    case Tint8:
    case Tuns8:
    case Tchar:
        return (const llvm::Type*)llvm::Type::Int8Ty;
    case Tint16:
    case Tuns16:
    case Twchar:
        return (const llvm::Type*)llvm::Type::Int16Ty;
    case Tint32:
    case Tuns32:
    case Tdchar:
        return (const llvm::Type*)llvm::Type::Int32Ty;
    case Tint64:
    case Tuns64:
        return (const llvm::Type*)llvm::Type::Int64Ty;

    case Tbool:
        return (const llvm::Type*)llvm::ConstantInt::getTrue()->getType();

    // floats
    case Tfloat32:
    case Timaginary32:
        return llvm::Type::FloatTy;
    case Tfloat64:
    case Timaginary64:
    case Tfloat80:
    case Timaginary80:
        return llvm::Type::DoubleTy;

    // complex
    case Tcomplex32:
    case Tcomplex64:
    case Tcomplex80:
        assert(0 && "complex number types not yet implemented");

    // pointers
    case Tpointer: {
        assert(t->next);
        if (t->next->ty == Tvoid)
            return (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
        else
            return (const llvm::Type*)llvm::PointerType::get(LLVM_DtoType(t->next));
    }

    // arrays
    case Tarray:
        return LLVM_DtoArrayType(t);
    case Tsarray:
        return LLVM_DtoStaticArrayType(t);

    // void
    case Tvoid:
        return llvm::Type::VoidTy;

    // aggregates
    case Tstruct:    {
        if (t->llvmType == 0)
        {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IRStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == i->type)
                    {
                        return i->recty.get();
                    }
                }
            }

            // forward declaration
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts->sym);
            ts->sym->toObjFile();
        }
        return t->llvmType;
    }

    case Tclass:    {
        if (t->llvmType == 0)
        {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IRStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == i->type)
                    {
                        return llvm::PointerType::get(i->recty.get());
                    }
                }
            }

            // forward declaration
            TypeClass* tc = (TypeClass*)t;
            assert(tc->sym);
            tc->sym->toObjFile();
        }
        return llvm::PointerType::get(t->llvmType);
    }

    // functions
    case Tfunction:
    {
        if (t->llvmType == 0) {
            return LLVM_DtoFunctionType(t);
        }
        else {
            return t->llvmType;
        }
    }

    // delegates
    case Tdelegate:
    {
        if (t->llvmType == 0) {
            return LLVM_DtoDelegateType(t);
        }
        else {
            return t->llvmType;
        }
    }

    // typedefs
    // enum
    case Ttypedef:
    case Tenum:
    {
        Type* bt = t->toBasetype();
        assert(bt);
        return LLVM_DtoType(bt);
    }

    default:
        printf("trying to convert unknown type with value %d\n", t->ty);
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* LLVM_DtoFunctionType(Type* t, const llvm::Type* thisparam)
{
    TypeFunction* f = (TypeFunction*)t;

    // parameter types
    const llvm::Type* rettype;
    std::vector<const llvm::Type*> paramvec;

    if (LLVM_DtoIsPassedByRef(f->next)) {
        rettype = llvm::PointerType::get(LLVM_DtoType(f->next));
        paramvec.push_back(rettype);
        rettype = llvm::Type::VoidTy;
    }
    else {
        Type* rt = f->next;
        if (rt)
        rettype = LLVM_DtoType(rt);
        else
        assert(0);
    }

    if (thisparam) {
        paramvec.push_back(thisparam);
    }

    size_t n = Argument::dim(f->parameters);
    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type;
        assert(argT);
        paramvec.push_back(LLVM_DtoType(argT));
    }

    Logger::cout() << "Return type: " << *rettype << '\n';

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, paramvec, f->varargs);
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static const llvm::FunctionType* LLVM_DtoVaFunctionType(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    const llvm::PointerType* i8pty = llvm::PointerType::get(llvm::Type::Int8Ty);
    std::vector<const llvm::Type*> args;

    if (fdecl->llvmInternal == LLVMva_start) {
        args.push_back(i8pty);
    }
    else if (fdecl->llvmInternal == LLVMva_intrinsic) {
        size_t n = Argument::dim(f->parameters);
        for (size_t i=0; i<n; ++i) {
            args.push_back(i8pty);
        }
    }
    else
    assert(0);

    const llvm::FunctionType* fty = llvm::FunctionType::get(llvm::Type::VoidTy, args, false);
    f->llvmType = fty;
    return fty;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl)
{
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        return LLVM_DtoVaFunctionType(fdecl);
    }

    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    // type has already been resolved
    if (f->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(f->llvmType);
    }

    bool typesafeVararg = false;
    if (f->linkage == LINKd && f->varargs == 1) {
        assert(fdecl->v_arguments);
        Logger::println("v_arguments = %s", fdecl->v_arguments->toChars());
        assert(fdecl->v_arguments->isParameter());
        typesafeVararg = true;
    }

    // return value type
    const llvm::Type* rettype;
    const llvm::Type* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    if (fdecl->isMain()) {
        rettype = llvm::Type::Int32Ty;
        actualRettype = rettype;
    }
    else if (rt) {
        if (LLVM_DtoIsPassedByRef(rt)) {
            rettype = llvm::PointerType::get(LLVM_DtoType(rt));
            actualRettype = llvm::Type::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = LLVM_DtoType(rt);
            actualRettype = rettype;
        }
    }
    else {
        assert(0);
    }

    // parameter types
    std::vector<const llvm::Type*> paramvec;

    if (retinptr) {
        Logger::cout() << "returning through pointer parameter: " << *rettype << '\n';
        paramvec.push_back(rettype);
    }

    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember()) {
            Logger::print("isMember = this is: %s\n", ad->type->toChars());
            const llvm::Type* thisty = LLVM_DtoType(ad->type);
            Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (llvm::isa<llvm::StructType>(thisty) || thisty == gIR->topstruct().recty.get())
                thisty = llvm::PointerType::get(thisty);
            paramvec.push_back(thisty);
            usesthis = true;
        }
        else
        assert(0);
    }
    else if (fdecl->isNested()) {
        paramvec.push_back(llvm::PointerType::get(llvm::Type::Int8Ty));
        usesthis = true;
    }

    if (typesafeVararg) {
        ClassDeclaration* ti = Type::typeinfo;
        if (!ti->llvmInitZ)
            ti->toObjFile();
        assert(ti->llvmInitZ);
        std::vector<const llvm::Type*> types;
        types.push_back(LLVM_DtoSize_t());
        types.push_back(llvm::PointerType::get(llvm::PointerType::get(ti->llvmInitZ->getType())));
        const llvm::Type* t1 = llvm::StructType::get(types);
        paramvec.push_back(llvm::PointerType::get(t1));
        paramvec.push_back(llvm::PointerType::get(llvm::Type::Int8Ty));
    }

    size_t n = Argument::dim(f->parameters);

    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = LLVM_DtoDType(arg->type);
        assert(argT);

        if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
            //assert(arg->vardecl);
            //arg->vardecl->refparam = true;
        }
        else
            arg->llvmCopy = true;

        const llvm::Type* at = LLVM_DtoType(argT);
        if (llvm::isa<llvm::StructType>(at)) {
            Logger::println("struct param");
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::ArrayType>(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(llvm::PointerType::get(at->getContainedType(0)));
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            if (argT->ty == Tstruct || argT->ty == Tclass)
                paramvec.push_back(llvm::PointerType::get(at));
            else
            assert(0);
        }
        /*if (llvm::isa<llvm::StructType>(at) || argT->ty == Tstruct || argT->ty == Tsarray) {
            paramvec.push_back(llvm::PointerType::get(at));
        }*/
        else {
            if (!arg->llvmCopy) {
                Logger::println("ref param");
                at = llvm::PointerType::get(at);
            }
            else {
                Logger::println("in param");
            }
            paramvec.push_back(at);
        }
    }

    // construct function type
    bool isvararg = !typesafeVararg && f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

    f->llvmType = functype;
    f->llvmRetInPtr = retinptr;
    f->llvmUsesThis = usesthis;
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* LLVM_DtoDelegateType(Type* t)
{
    const llvm::Type* i8ptr = llvm::PointerType::get(llvm::Type::Int8Ty);
    const llvm::Type* func = LLVM_DtoFunctionType(t->next, i8ptr);
    const llvm::Type* funcptr = llvm::PointerType::get(func);

    std::vector<const llvm::Type*> types;
    types.push_back(i8ptr);
    types.push_back(funcptr);
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::Type* LLVM_DtoStructType(Type* t)
{
    assert(0);
    std::vector<const llvm::Type*> types;
    return llvm::StructType::get(types);
}


//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* LLVM_DeclareMemIntrinsic(const char* name, int bits, bool set=false)
{
    assert(bits == 32 || bits == 64);
    const llvm::Type* int8ty =    (const llvm::Type*)llvm::Type::Int8Ty;
    const llvm::Type* int32ty =   (const llvm::Type*)llvm::Type::Int32Ty;
    const llvm::Type* int64ty =   (const llvm::Type*)llvm::Type::Int64Ty;
    const llvm::Type* int8ptrty = (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
    const llvm::Type* voidty =    (const llvm::Type*)llvm::Type::VoidTy;

    assert(gIR);
    assert(gIR->module);

    // parameter types
    std::vector<const llvm::Type*> pvec;
    pvec.push_back(int8ptrty);
    pvec.push_back(set?int8ty:int8ptrty);
    pvec.push_back(bits==32?int32ty:int64ty);
    pvec.push_back(int32ty);
    llvm::FunctionType* functype = llvm::FunctionType::get(voidty, pvec, false);
    return new llvm::Function(functype, llvm::GlobalValue::ExternalLinkage, name, gIR->module);
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memset.i32
llvm::Function* LLVM_DeclareMemSet32()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memset.i32", 32, true);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_DeclareMemSet64()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memset.i64", 64, true);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i32
llvm::Function* LLVM_DeclareMemCpy32()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memcpy.i32", 32);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i64
llvm::Function* LLVM_DeclareMemCpy64()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memcpy.i64", 64);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoStructZeroInit(llvm::Value* v)
{
    assert(gIR);
    uint64_t n = gTargetData->getTypeSize(v->getType()->getContainedType(0));
    //llvm::Type* sarrty = llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
    llvm::Type* sarrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* sarr = new llvm::BitCastInst(v,sarrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = sarr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoStructCopy(llvm::Value* dst, llvm::Value* src)
{
    assert(dst->getType() == src->getType());
    assert(gIR);

    uint64_t n = gTargetData->getTypeSize(dst->getType()->getContainedType(0));
    //llvm::Type* sarrty = llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
    llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    return new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* LLVM_DtoConstStructInitializer(StructInitializer* si)
{
    llvm::StructType* structtype = llvm::cast<llvm::StructType>(si->ad->llvmType);
    size_t n = structtype->getNumElements();

    assert(si->value.dim == si->vars.dim);

    std::vector<llvm::Constant*> inits;
    inits.resize(n, NULL);
    for (int i = 0; i < si->value.dim; ++i)
    {
        Initializer* ini = (Initializer*)si->value.data[i];
        assert(ini);

        VarDeclaration* vd = (VarDeclaration*)si->vars.data[i];
        Type* vdtype = LLVM_DtoDType(vd->type);
        assert(vd);
        Logger::println("vars[%d] = %s", i, vd->toChars());

        std::vector<unsigned> idxs;
        si->ad->offsetToIndex(vdtype, vd->offset, idxs);
        assert(idxs.size() == 1);
        unsigned idx = idxs[0];

        llvm::Constant* v = 0;

        if (ExpInitializer* ex = ini->isExpInitializer())
        {
            v = ex->exp->toConstElem(gIR);
        }
        else if (StructInitializer* si = ini->isStructInitializer())
        {
            v = LLVM_DtoConstStructInitializer(si);
        }
        else if (ArrayInitializer* ai = ini->isArrayInitializer())
        {
            v = LLVM_DtoConstArrayInitializer(ai);
        }
        else if (ini->isVoidInitializer())
        {
            v = llvm::UndefValue::get(structtype->getElementType(idx));
        }
        else
        assert(v);

        inits[idx] = v;
    }

    // fill out nulls
    assert(si->ad->llvmInitZ);
    if (si->ad->llvmInitZ->isNullValue())
    {
        for (int i = 0; i < n; ++i)
        {
            if (inits[i] == 0)
            {
                inits[i] = llvm::Constant::getNullValue(structtype->getElementType(i));
            }
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            if (inits[i] == 0)
            {
                inits[i] = si->ad->llvmInitZ->getOperand(i);
            }
        }
    }

    return llvm::ConstantStruct::get(structtype, inits);
}



//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoNullDelegate(llvm::Value* v)
{
    assert(gIR);
    d_uns64 n = (global.params.is64bit) ? 16 : 8;

    llvm::Type* i8p_ty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* arr = new llvm::BitCastInst(v,i8p_ty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = arr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoDelegateCopy(llvm::Value* dst, llvm::Value* src)
{
    assert(dst->getType() == src->getType());
    assert(gIR);

    d_uns64 n = (global.params.is64bit) ? 16 : 8;

    llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    return new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoCompareDelegate(TOK op, llvm::Value* lhs, llvm::Value* rhs)
{
    llvm::ICmpInst::Predicate pred = (op == TOKequal) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    llvm::Value* l = gIR->ir->CreateLoad(LLVM_DtoGEPi(lhs,0,0,"tmp"),"tmp");
    llvm::Value* r = gIR->ir->CreateLoad(LLVM_DtoGEPi(rhs,0,0,"tmp"),"tmp");
    llvm::Value* b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    l = gIR->ir->CreateLoad(LLVM_DtoGEPi(lhs,0,1,"tmp"),"tmp");
    r = gIR->ir->CreateLoad(LLVM_DtoGEPi(rhs,0,1,"tmp"),"tmp");
    llvm::Value* b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    llvm::Value* b = gIR->ir->CreateAnd(b1,b2,"tmp");
    if (op == TOKnotequal)
        return gIR->ir->CreateNot(b,"tmp");
    return b;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalValue::LinkageTypes LLVM_DtoLinkage(PROT prot, uint stc)
{
    switch(prot)
    {
    case PROTprivate:
        return llvm::GlobalValue::InternalLinkage;

    case PROTpublic:
    case PROTpackage:
    case PROTprotected:
    case PROTexport:
        return llvm::GlobalValue::ExternalLinkage;

    case PROTundefined:
    case PROTnone:
        assert(0 && "Unsupported linkage type");
    }
    return llvm::GlobalValue::ExternalLinkage;

/*      ExternalLinkage = 0, LinkOnceLinkage, WeakLinkage, AppendingLinkage,
  InternalLinkage, DLLImportLinkage, DLLExportLinkage, ExternalWeakLinkage,
  GhostLinkage */
}

//////////////////////////////////////////////////////////////////////////////////////////

unsigned LLVM_DtoCallingConv(LINK l)
{
    if (l == LINKc)
        return llvm::CallingConv::C;
    else if (l == LINKd || l == LINKdefault)
        return llvm::CallingConv::Fast;
    else if (l == LINKwindows)
        return llvm::CallingConv::X86_StdCall;
    else
        assert(0 && "Unsupported calling convention");
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoPointedType(llvm::Value* ptr, llvm::Value* val)
{
    const llvm::Type* ptrTy = ptr->getType()->getContainedType(0);
    const llvm::Type* valTy = val->getType();
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
        const llvm::IntegerType* pt = llvm::cast<const llvm::IntegerType>(ptrTy);
        const llvm::IntegerType* vt = llvm::cast<const llvm::IntegerType>(valTy);
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

llvm::Value* LLVM_DtoBoolean(llvm::Value* val)
{
    const llvm::Type* t = val->getType();
    if (t->isInteger())
    {
        if (t == llvm::Type::Int1Ty)
            return val;
        else {
            llvm::Value* zero = llvm::ConstantInt::get(t, 0, false);
            return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, val, zero, "tmp", gIR->scopebb());
        }
    }
    else if (llvm::isa<llvm::PointerType>(t)) {
        const llvm::Type* st = LLVM_DtoSize_t();
        llvm::Value* ptrasint = new llvm::PtrToIntInst(val,st,"tmp",gIR->scopebb());
        llvm::Value* zero = llvm::ConstantInt::get(st, 0, false);
        return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, ptrasint, zero, "tmp", gIR->scopebb());
    }
    else
    {
        Logger::cout() << *t << '\n';
    }
    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::Type* LLVM_DtoSize_t()
{
    if (global.params.is64bit)
    return llvm::Type::Int64Ty;
    else
    return llvm::Type::Int32Ty;
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoMain()
{
    // emit main function llvm style
    // int main(int argc, char**argv, char**env);

    assert(gIR != 0);
    IRState& ir = *gIR;

    assert(ir.emitMain && ir.mainFunc);

    // parameter types
    std::vector<const llvm::Type*> pvec;
    pvec.push_back((const llvm::Type*)llvm::Type::Int32Ty);
    const llvm::Type* chPtrType = (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    const llvm::Type* rettype = (const llvm::Type*)llvm::Type::Int32Ty;

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, pvec, false);
    llvm::Function* func = new llvm::Function(functype,llvm::GlobalValue::ExternalLinkage,"main",ir.module);

    llvm::BasicBlock* bb = new llvm::BasicBlock("entry",func);

    // call static ctors
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(ir.module,"_d_run_module_ctors");
    new llvm::CallInst(fn,"",bb);

    // call user main function
    llvm::CallInst* call = new llvm::CallInst(ir.mainFunc,"ret",bb);
    call->setCallingConv(ir.mainFunc->getCallingConv());

    // call static dtors
    fn = LLVM_D_GetRuntimeFunction(ir.module,"_d_run_module_dtors");
    new llvm::CallInst(fn,"",bb);

    // return
    new llvm::ReturnInst(call,bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoCallClassDtors(TypeClass* tc, llvm::Value* instance)
{
    Array* arr = &tc->sym->dtors;
    for (size_t i=0; i<arr->dim; i++)
    {
        FuncDeclaration* fd = (FuncDeclaration*)arr->data[i];
        assert(fd->llvmValue);
        new llvm::CallInst(fd->llvmValue, instance, "", gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoInitClass(TypeClass* tc, llvm::Value* dst)
{
    assert(gIR);

    assert(tc->llvmType);
    uint64_t size_t_size = gTargetData->getTypeSize(LLVM_DtoSize_t());
    uint64_t n = gTargetData->getTypeSize(tc->llvmType) - size_t_size;

    // set vtable field
    llvm::Value* vtblvar = LLVM_DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
    assert(tc->sym->llvmVtbl);
    new llvm::StoreInst(tc->sym->llvmVtbl, vtblvar, gIR->scopebb());

    // copy the static initializer
    if (n > 0) {
        assert(tc->llvmInit);
        assert(dst->getType() == tc->llvmInit->getType());

        llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

        llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
        dstarr = LLVM_DtoGEPi(dstarr,size_t_size,"tmp",gIR->scopebb());

        llvm::Value* srcarr = new llvm::BitCastInst(tc->llvmInit,arrty,"tmp",gIR->scopebb());
        srcarr = LLVM_DtoGEPi(srcarr,size_t_size,"tmp",gIR->scopebb());

        llvm::Function* fn = LLVM_DeclareMemCpy32();
        std::vector<llvm::Value*> llargs;
        llargs.resize(4);
        llargs[0] = dstarr;
        llargs[1] = srcarr;
        llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
        llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

        new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* LLVM_DtoConstInitializer(Type* type, Initializer* init)
{
    llvm::Constant* _init = 0; // may return zero
    if (!init)
    {
        Logger::println("const default initializer for %s", type->toChars());
        _init = type->defaultInit()->toConstElem(gIR);
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("const expression initializer");
        _init = ex->exp->toConstElem(gIR);
    }
    else if (StructInitializer* si = init->isStructInitializer())
    {
        Logger::println("const struct initializer");
        _init = LLVM_DtoConstStructInitializer(si);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        Logger::println("const array initializer");
        _init = LLVM_DtoConstArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        Logger::println("const void initializer");
        const llvm::Type* ty = LLVM_DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported const initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoInitializer(Initializer* init)
{
    if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("expression initializer");
        elem* e = ex->exp->toElem(gIR);
        delete e;
    }
    else {
        Logger::println("unsupported initializer: %s", init->toChars());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = i0;
    v[1] = i1;
    Logger::cout() << "DtoGEP: " << *ptr << '\n';
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb)
{
    size_t n = src.size();
    std::vector<llvm::Value*> dst(n);
    std::ostream& ostr = Logger::cout();
    ostr << "indices for '" << *ptr << "':";
    for (size_t i=0; i<n; ++i)
    {
        ostr << ' ' << i;
        dst[i] = llvm::ConstantInt::get(llvm::Type::Int32Ty, src[i], false);
    }
    ostr << '\n';
    return new llvm::GetElementPtrInst(ptr, dst.begin(), dst.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i, const std::string& var, llvm::BasicBlock* bb)
{
    return new llvm::GetElementPtrInst(ptr, llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i0, false);
    v[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i1, false);
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* LLVM_DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)LLVM_DtoDType(fdecl->type);
    const llvm::FunctionType* fty = LLVM_DtoVaFunctionType(fdecl);
    llvm::Constant* fn = 0;

    if (fdecl->llvmInternal == LLVMva_start) {
        fn = gIR->module->getOrInsertFunction("llvm.va_start", fty);
        assert(fn);
    }
    else if (fdecl->llvmInternal == LLVMva_intrinsic) {
        fn = gIR->module->getOrInsertFunction(fdecl->llvmInternal1, fty);
        assert(fn);
    }
    else
    assert(0);

    llvm::Function* func = llvm::cast_or_null<llvm::Function>(fn);
    assert(func);
    assert(func->isIntrinsic());
    fdecl->llvmValue = func;
    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_DtoDeclareFunction(FuncDeclaration* fdecl)
{
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        return LLVM_DtoDeclareVaFunction(fdecl);
    }

    // mangled name
    char* mangled_name;
    if (fdecl->llvmInternal == LLVMintrinsic)
        mangled_name = fdecl->llvmInternal1;
    else
        mangled_name = fdecl->mangle();

    // unit test special handling
    if (fdecl->isUnitTestDeclaration())
    {
        assert(0 && "no unittests yet");
        /*const llvm::FunctionType* fnty = llvm::FunctionType::get(llvm::Type::VoidTy, std::vector<const llvm::Type*>(), false);
        // make the function
        llvm::Function* func = gIR->module->getFunction(mangled_name);
        if (func == 0)
            func = new llvm::Function(fnty,llvm::GlobalValue::InternalLinkage,mangled_name,gIR->module);
        func->setCallingConv(llvm::CallingConv::Fast);
        fdecl->llvmValue = func;
        return func;
        */
    }

    // regular function
    TypeFunction* f = (TypeFunction*)LLVM_DtoDType(fdecl->type);
    assert(f != 0);

    if (fdecl->llvmValue != 0) {
        if (!llvm::isa<llvm::Function>(fdecl->llvmValue))
        {
            Logger::cout() << *fdecl->llvmValue << '\n';
            assert(0);
        }
        return llvm::cast<llvm::Function>(fdecl->llvmValue);
    }

    Logger::print("FuncDeclaration::toObjFile(%s): %s\n", fdecl->needThis()?"this":"static",fdecl->toChars());
    LOG_SCOPE;

    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error("intrinsics cannot have function bodies");
        fatal();
    }

    // construct function
    const llvm::FunctionType* functype = (f->llvmType == 0) ? LLVM_DtoFunctionType(fdecl) : llvm::cast<llvm::FunctionType>(f->llvmType);

    // make the function
    llvm::Function* func = gIR->module->getFunction(mangled_name);
    if (func == 0) {
        func = new llvm::Function(functype,LLVM_DtoLinkage(fdecl->protection, fdecl->storage_class),mangled_name,gIR->module);
    }

    if (fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(LLVM_DtoCallingConv(f->linkage));

    fdecl->llvmValue = func;
    f->llvmType = functype;
    assert(llvm::isa<llvm::FunctionType>(f->llvmType));

    if (fdecl->isMain()) {
        gIR->mainFunc = func;
    }

    // name parameters
    llvm::Function::arg_iterator iarg = func->arg_begin();
    int k = 0;
    if (f->llvmRetInPtr) {
        iarg->setName("retval");
        f->llvmRetArg = iarg;
        ++iarg;
    }
    if (f->llvmUsesThis) {
        iarg->setName("this");
        ++iarg;
    }
    int varargs = -1;
    if (f->linkage == LINKd && f->varargs == 1)
        varargs = 0;
    for (; iarg != func->arg_end(); ++iarg)
    {
        Argument* arg = Argument::getNth(f->parameters, k++);
        //arg->llvmValue = iarg;
        //printf("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
        if (arg && arg->ident != 0) {
            if (arg->vardecl) {
                arg->vardecl->llvmValue = iarg;
            }
            iarg->setName(arg->ident->toChars());
        }
        else if (!arg && varargs >= 0) {
            if (varargs == 0) {
                iarg->setName("_arguments");
                fdecl->llvmArguments = iarg;
            }
            else if (varargs == 1) {
                iarg->setName("_argptr");
                fdecl->llvmArgPtr = iarg;
            }
            else
            assert(0);
            varargs++;
        }
        else {
            iarg->setName("unnamed");
        }
    }

    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoGiveArgumentStorage(elem* l)
{
    assert(l->mem == 0);
    assert(l->val);
    assert(llvm::isa<llvm::Argument>(l->val));
    assert(l->vardecl != 0);

    llvm::AllocaInst* allocainst = new llvm::AllocaInst(l->val->getType(), l->val->getName()+"_storage", gIR->topallocapoint());
    l->mem = allocainst;
    l->vardecl->llvmValue = l->mem;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoRealloc(llvm::Value* ptr, const llvm::Type* ty)
{
    /*size_t sz = gTargetData->getTypeSize(ty);
    llvm::ConstantInt* n = llvm::ConstantInt::get(LLVM_DtoSize_t(), sz, false);
    if (ptr == 0) {
        llvm::PointerType* i8pty = llvm::PointerType::get(llvm::Type::Int8Ty);
        ptr = llvm::ConstantPointerNull::get(i8pty);
    }
    return LLVM_DtoRealloc(ptr, n);*/
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoRealloc(llvm::Value* ptr, llvm::Value* n)
{
    assert(ptr);
    assert(n);

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_realloc");
    assert(fn);

    llvm::Value* newptr = ptr;

    llvm::PointerType* i8pty = llvm::PointerType::get(llvm::Type::Int8Ty);
    if (ptr->getType() != i8pty) {
        newptr = new llvm::BitCastInst(ptr,i8pty,"tmp",gIR->scopebb());
    }

    std::vector<llvm::Value*> args;
    args.push_back(newptr);
    args.push_back(n);
    llvm::Value* ret = new llvm::CallInst(fn, args.begin(), args.end(), "tmprealloc", gIR->scopebb());

    return ret->getType() == ptr->getType() ? ret : new llvm::BitCastInst(ret,ptr->getType(),"tmp",gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoAssert(llvm::Value* cond, llvm::Value* loc, llvm::Value* msg)
{
    assert(loc);
    std::vector<llvm::Value*> llargs;
    llargs.resize(3);
    llargs[0] = cond ? LLVM_DtoBoolean(cond) : llvm::ConstantInt::getFalse();
    llargs[1] = loc;
    llargs[2] = msg ? msg : llvm::ConstantPointerNull::get(llvm::PointerType::get(llvm::Type::Int8Ty));

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_assert");
    assert(fn);
    llvm::CallInst* call = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
    call->setCallingConv(llvm::CallingConv::C);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoArgument(const llvm::Type* paramtype, Argument* fnarg, Expression* argexp)
{
    llvm::Value* retval = 0;

    bool haslvals = !gIR->exps.empty();
    if (haslvals)
        gIR->exps.push_back(IRExp(NULL,NULL,NULL));

    elem* arg = argexp->toElem(gIR);

    if (haslvals)
        gIR->exps.pop_back();

    if (arg->inplace) {
        assert(arg->mem != 0);
        retval = arg->mem;
        delete arg;
        return retval;
    }

    Type* realtype = LLVM_DtoDType(argexp->type);
    TY argty = realtype->ty;
    if (LLVM_DtoIsPassedByRef(realtype)) {
        if (!fnarg || !fnarg->llvmCopy) {
            retval = arg->getValue();
            assert(retval != 0);
        }
        else {
            llvm::Value* allocaInst = 0;
            llvm::BasicBlock* entryblock = &gIR->topfunc()->front();
            //const llvm::PointerType* pty = llvm::cast<llvm::PointerType>(arg->mem->getType());
            const llvm::Type* realtypell = LLVM_DtoType(realtype);
            const llvm::PointerType* pty = llvm::PointerType::get(realtypell);
            if (argty == Tstruct) {
                allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                LLVM_DtoStructCopy(allocaInst,arg->mem);
            }
            else if (argty == Tdelegate) {
                allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                LLVM_DtoDelegateCopy(allocaInst,arg->mem);
            }
            else if (argty == Tarray) {
                if (arg->type == elem::SLICE) {
                    allocaInst = new llvm::AllocaInst(realtypell, "tmpparam", gIR->topallocapoint());
                    LLVM_DtoSetArray(allocaInst, arg->arg, arg->mem);
                }
                else {
                    allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                    LLVM_DtoArrayAssign(allocaInst,arg->mem);
                }
            }
            else
            assert(0);

            assert(allocaInst != 0);
            retval = allocaInst;
        }
    }
    else if (!fnarg || fnarg->llvmCopy) {
        Logger::println("regular arg");
        assert(arg->type != elem::SLICE);
        if (arg->mem) Logger::cout() << "mem = " << *arg->mem << '\n';
        if (arg->val) Logger::cout() << "val = " << *arg->val << '\n';
        if (arg->arg) Logger::cout() << "arg = " << *arg->arg << '\n';
        retval = arg->arg ? arg->arg : arg->field ? arg->mem : arg->getValue();
    }
    else {
        Logger::println("as ptr arg");
        retval = arg->mem ? arg->mem : arg->val;
        if (paramtype && retval->getType() != paramtype)
        {
            assert(retval->getType() == paramtype->getContainedType(0));
            LLVM_DtoGiveArgumentStorage(arg);
            new llvm::StoreInst(retval, arg->mem, gIR->scopebb());
            retval = arg->mem;
        }
    }

    delete arg;

    if (fnarg && paramtype && retval->getType() != paramtype) {
        Logger::cout() << "got '" << *retval->getType() << "' expected '" << *paramtype << "'\n";
        assert(0 && "parameter type that was actually passed is invalid");
    }
    return retval;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoNestedVariable(VarDeclaration* vd)
{
    FuncDeclaration* fd = vd->toParent()->isFuncDeclaration();
    assert(fd != NULL);

    IRFunction* fcur = &gIR->func();
    FuncDeclaration* f = fcur->decl;

    // on this stack
    if (fd == f) {
        return LLVM_DtoGEPi(vd->llvmValue,0,unsigned(vd->llvmNestedIndex),"tmp");
    }

    // on a caller stack
    llvm::Value* ptr = f->llvmThisVar;
    assert(ptr);

    f = f->toParent()->isFuncDeclaration();
    assert(f);
    assert(f->llvmNested);
    const llvm::Type* nesttype = f->llvmNested->getType();
    assert(nesttype);

    ptr = gIR->ir->CreateBitCast(ptr, nesttype, "tmp");

    Logger::cout() << "nested var reference:" << '\n' << *ptr << *nesttype << '\n';

    while (f) {
        if (fd == f) {
            return LLVM_DtoGEPi(ptr,0,vd->llvmNestedIndex,"tmp");
        }
        else {
            ptr = LLVM_DtoGEPi(ptr,0,0,"tmp");
            ptr = gIR->ir->CreateLoad(ptr,"tmp");
        }
        f = f->toParent()->isFuncDeclaration();
    }

    assert(0 && "nested var not found");
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoAssign(Type* t, llvm::Value* lhs, llvm::Value* rhs)
{
    Logger::cout() << "assignment:" << '\n' << *lhs << *rhs << '\n';

    if (t->ty == Tstruct) {
        assert(lhs->getType() == rhs->getType());
        LLVM_DtoStructCopy(lhs,rhs);
    }
    else if (t->ty == Tarray) {
        assert(lhs->getType() == rhs->getType());
        LLVM_DtoArrayAssign(lhs,rhs);
    }
    else if (t->ty == Tsarray) {
        assert(lhs->getType() == rhs->getType());
        LLVM_DtoStaticArrayCopy(lhs,rhs);
    }
    else if (t->ty == Tdelegate) {
        assert(lhs->getType() == rhs->getType());
        LLVM_DtoDelegateCopy(lhs,rhs);
    }
    else {
        assert(lhs->getType()->getContainedType(0) == rhs->getType());
        gIR->ir->CreateStore(rhs, lhs);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::ConstantInt* LLVM_DtoConstSize_t(size_t i)
{
    return llvm::ConstantInt::get(LLVM_DtoSize_t(), i, false);
}
llvm::ConstantInt* LLVM_DtoConstUint(unsigned i)
{
    return llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* LLVM_DtoConstString(const char* str)
{
    std::string s(str);
    llvm::Constant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, "stringliteral", gIR->module);
    llvm::Constant* idxs[2] = { LLVM_DtoConstUint(0), LLVM_DtoConstUint(0) };
    return LLVM_DtoConstantSlice(
        LLVM_DtoConstSize_t(s.length()),
        llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2)
    );
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes)
{
    assert(dst->getType() == src->getType());

    llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);
    llvm::Value *dstarr, *srcarr;
    if (dst->getType() == arrty)
    {
        dstarr = dst;
        srcarr = src;
    }
    else
    {
        dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
        srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());
    }

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemCpy64() : LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = nbytes;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}
