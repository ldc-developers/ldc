#include <iostream>

#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/CallingConv.h"

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
        return llvm::Type::FloatTy;
    case Tfloat64:
    case Tfloat80:
        return llvm::Type::DoubleTy;

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
    case Ttypedef:
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

    TY retty = f->next->ty;

    if (retty == Tstruct || retty == Tdelegate || retty == Tarray) {
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

const llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    // type has already been resolved
    if (f->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(f->llvmType);
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
        if (rt->ty == Tstruct || rt->ty == Tdelegate || rt->ty == Tarray) {
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

    size_t n = Argument::dim(f->parameters);
    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type;
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
    bool isvararg = f->varargs;
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

llvm::Value* LLVM_DtoStructZeroInit(TypeStruct* t, llvm::Value* v)
{
    assert(gIR);
    uint64_t n = gTargetData->getTypeSize(t->llvmType);
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

llvm::Value* LLVM_DtoStructCopy(TypeStruct* t, llvm::Value* dst, llvm::Value* src)
{
    assert(dst->getType() == src->getType());
    assert(gIR);

    uint64_t n = gTargetData->getTypeSize(t->llvmType);
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
llvm::Constant* LLVM_DtoStructInitializer(StructInitializer* si)
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
        assert(vd);
        Logger::println("vars[%d] = %s", i, vd->toChars());

        std::vector<unsigned> idxs;
        si->ad->offsetToIndex(vd->type, vd->offset, idxs);
        assert(idxs.size() == 1);
        unsigned idx = idxs[0];

        llvm::Constant* v = 0;

        if (ExpInitializer* ex = ini->isExpInitializer())
        {
            elem* e = ex->exp->toElem(gIR);
            v = llvm::cast<llvm::Constant>(e->val);
            delete e;
        }
        else if (StructInitializer* si = ini->isStructInitializer())
        {
            v = LLVM_DtoStructInitializer(si);
        }
        else if (ArrayInitializer* ai = ini->isArrayInitializer())
        {
            v = LLVM_DtoArrayInitializer(ai);
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

    /*
    // return value type
    const llvm::Type* rettype;
    Type* rt = f->next;
    if (rt) {
        rettype = LLVM_DtoType(rt);
    }
    else {
        assert(0);
    }

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, paramvec, false);
    */
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

llvm::Constant* LLVM_DtoInitializer(Type* type, Initializer* init)
{
    llvm::Constant* _init = 0; // may return zero
    if (!init)
    {
        Logger::println("default initializer");
        elem* e = type->defaultInit()->toElem(gIR);
        if (!e->inplace && !e->isNull()) {
            _init = llvm::cast<llvm::Constant>(e->getValue());
        }
        delete e;
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("expression initializer");
        elem* e = ex->exp->toElem(gIR);
        if (!e->inplace && !e->isNull()) {
            _init = llvm::cast<llvm::Constant>(e->getValue());
        }
        delete e;
    }
    else if (StructInitializer* si = init->isStructInitializer())
    {
        Logger::println("struct initializer");
        _init = LLVM_DtoStructInitializer(si);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        Logger::println("array initializer");
        _init = LLVM_DtoArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        Logger::println("void initializer");
        const llvm::Type* ty = LLVM_DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = i0;
    v[1] = i1;
    Logger::cout() << "DtoGEP: " << *ptr << '\n';
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb)
{
    size_t n = src.size();
    std::vector<llvm::Value*> dst(n);
    Logger::cout() << "indices:";
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << ' ' << i;
        dst[i] = llvm::ConstantInt::get(llvm::Type::Int32Ty, src[i], false);
    }
    Logger::cout() << '\n';
    return new llvm::GetElementPtrInst(ptr, dst.begin(), dst.end(), var, bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i, const std::string& var, llvm::BasicBlock* bb)
{
    return new llvm::GetElementPtrInst(ptr, llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false), var, bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i0, false);
    v[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i1, false);
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_DtoDeclareFunction(FuncDeclaration* fdecl)
{
    // mangled name
    char* mangled_name = (fdecl->llvmInternal == LLVMintrinsic) ? fdecl->llvmInternal1 : fdecl->mangle();

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
    TypeFunction* f = (TypeFunction*)fdecl->type;
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
    for (; iarg != func->arg_end(); ++iarg)
    {
        Argument* arg = Argument::getNth(f->parameters, k++);
        assert(arg != 0);
        //arg->llvmValue = iarg;
        //printf("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
        if (arg->ident != 0) {
            if (arg->vardecl) {
                arg->vardecl->llvmValue = iarg;
            }
            iarg->setName(arg->ident->toChars());
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







