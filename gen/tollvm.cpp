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
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/structs.h"

bool DtoIsPassedByRef(Type* type)
{
    TY t = DtoDType(type)->ty;
    return (t == Tstruct || t == Tarray || t == Tdelegate || t == Tsarray);
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

const llvm::Type* DtoType(Type* t)
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
        return DtoComplexType(llvm::Type::FloatTy);
    case Tcomplex64:
    case Tcomplex80:
        return DtoComplexType(llvm::Type::DoubleTy);

    // pointers
    case Tpointer: {
        assert(t->next);
        if (t->next->ty == Tvoid)
            return (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
        else
            return (const llvm::Type*)llvm::PointerType::get(DtoType(t->next));
    }

    // arrays
    case Tarray:
        return DtoArrayType(t);
    case Tsarray:
        return DtoStaticArrayType(t);

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
            return DtoFunctionType(t,NULL);
        }
        else {
            return t->llvmType;
        }
    }

    // delegates
    case Tdelegate:
    {
        if (t->llvmType == 0) {
            return DtoDelegateType(t);
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
        return DtoType(bt);
    }

    default:
        printf("trying to convert unknown type with value %d\n", t->ty);
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* DtoFunctionType(Type* type, const llvm::Type* thistype, bool ismain)
{
    TypeFunction* f = (TypeFunction*)type;
    assert(f != 0);

    bool typesafeVararg = false;
    if (f->linkage == LINKd && f->varargs == 1) {
        typesafeVararg = true;
    }

    // return value type
    const llvm::Type* rettype;
    const llvm::Type* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    if (ismain) {
        rettype = llvm::Type::Int32Ty;
        actualRettype = rettype;
    }
    else {
        assert(rt);
        if (DtoIsPassedByRef(rt)) {
            rettype = llvm::PointerType::get(DtoType(rt));
            actualRettype = llvm::Type::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = DtoType(rt);
            actualRettype = rettype;
        }
    }

    // parameter types
    std::vector<const llvm::Type*> paramvec;

    if (retinptr) {
        Logger::cout() << "returning through pointer parameter: " << *rettype << '\n';
        paramvec.push_back(rettype);
    }

    if (thistype) {
        paramvec.push_back(thistype);
        usesthis = true;
    }

    if (typesafeVararg) {
        ClassDeclaration* ti = Type::typeinfo;
        if (!ti->llvmInitZ)
            ti->toObjFile();
        assert(ti->llvmInitZ);
        std::vector<const llvm::Type*> types;
        types.push_back(DtoSize_t());
        types.push_back(llvm::PointerType::get(llvm::PointerType::get(ti->llvmInitZ->getType())));
        const llvm::Type* t1 = llvm::StructType::get(types);
        paramvec.push_back(llvm::PointerType::get(t1));
        paramvec.push_back(llvm::PointerType::get(llvm::Type::Int8Ty));
    }

    size_t n = Argument::dim(f->parameters);

    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = DtoDType(arg->type);
        assert(argT);

        if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
            //assert(arg->vardecl);
            //arg->vardecl->refparam = true;
        }
        else
            arg->llvmCopy = true;

        const llvm::Type* at = DtoType(argT);
        if (isaStruct(at)) {
            Logger::println("struct param");
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (isaArray(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(llvm::PointerType::get(at->getContainedType(0)));
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            assert(argT->ty == Tstruct || argT->ty == Tclass);
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
    bool isvararg = !typesafeVararg && f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

    f->llvmRetInPtr = retinptr;
    f->llvmUsesThis = usesthis;
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static const llvm::FunctionType* DtoVaFunctionType(FuncDeclaration* fdecl)
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

const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl)
{
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        return DtoVaFunctionType(fdecl);
    }

    // type has already been resolved
    if (fdecl->type->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->llvmType);
    }

    const llvm::Type* thisty = NULL;
    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember()) {
            Logger::print("isMember = this is: %s\n", ad->type->toChars());
            thisty = DtoType(ad->type);
            Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (isaStruct(thisty) || thisty == gIR->topstruct().recty.get())
                thisty = llvm::PointerType::get(thisty);
        }
        else
        assert(0);
    }
    else if (fdecl->isNested()) {
        thisty = llvm::PointerType::get(llvm::Type::Int8Ty);
    }

    const llvm::FunctionType* functype = DtoFunctionType(fdecl->type, thisty, fdecl->isMain());
    fdecl->type->llvmType = functype;
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoDelegateType(Type* t)
{
    const llvm::Type* i8ptr = llvm::PointerType::get(llvm::Type::Int8Ty);
    const llvm::Type* func = DtoFunctionType(t->next, i8ptr);
    const llvm::Type* funcptr = llvm::PointerType::get(func);

    std::vector<const llvm::Type*> types;
    types.push_back(i8ptr);
    types.push_back(funcptr);
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoComplexType(const llvm::Type* base)
{
    std::vector<const llvm::Type*> types;
    types.push_back(base);
    types.push_back(base);
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

llvm::Value* DtoNullDelegate(llvm::Value* v)
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

llvm::Value* DtoDelegateCopy(llvm::Value* dst, llvm::Value* src)
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

llvm::Value* DtoCompareDelegate(TOK op, llvm::Value* lhs, llvm::Value* rhs)
{
    llvm::ICmpInst::Predicate pred = (op == TOKequal) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    llvm::Value* l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,0,"tmp"),"tmp");
    llvm::Value* r = gIR->ir->CreateLoad(DtoGEPi(rhs,0,0,"tmp"),"tmp");
    llvm::Value* b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,1,"tmp"),"tmp");
    r = gIR->ir->CreateLoad(DtoGEPi(rhs,0,1,"tmp"),"tmp");
    llvm::Value* b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    llvm::Value* b = gIR->ir->CreateAnd(b1,b2,"tmp");
    if (op == TOKnotequal)
        return gIR->ir->CreateNot(b,"tmp");
    return b;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalValue::LinkageTypes DtoLinkage(PROT prot, uint stc)
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

unsigned DtoCallingConv(LINK l)
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

llvm::Value* DtoPointedType(llvm::Value* ptr, llvm::Value* val)
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

llvm::Value* DtoBoolean(llvm::Value* val)
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
    else if (isaPointer(t)) {
        const llvm::Type* st = DtoSize_t();
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

const llvm::Type* DtoSize_t()
{
    if (global.params.is64bit)
    return llvm::Type::Int64Ty;
    else
    return llvm::Type::Int32Ty;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMain()
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
    llvm::Instruction* apt = new llvm::CallInst(fn,"",bb);

    // call user main function
    const llvm::FunctionType* mainty = ir.mainFunc->getFunctionType();
    llvm::CallInst* call;
    if (mainty->getNumParams() > 0)
    {
        // main with arguments
        assert(mainty->getNumParams() == 1);
        std::vector<llvm::Value*> args;
        llvm::Function* mfn = LLVM_D_GetRuntimeFunction(ir.module,"_d_main_args");

        llvm::Function::arg_iterator argi = func->arg_begin();
        args.push_back(argi++);
        args.push_back(argi++);

        const llvm::Type* at = mainty->getParamType(0)->getContainedType(0);
        llvm::Value* arr = new llvm::AllocaInst(at->getContainedType(1)->getContainedType(0), func->arg_begin(), "argstorage", apt);
        llvm::Value* a = new llvm::AllocaInst(at, "argarray", apt);
        llvm::Value* ptr = DtoGEPi(a,0,0,"tmp",bb);
        llvm::Value* v = args[0];
        if (v->getType() != DtoSize_t())
            v = new llvm::ZExtInst(v, DtoSize_t(), "tmp", bb);
        new llvm::StoreInst(v,ptr,bb);
        ptr = DtoGEPi(a,0,1,"tmp",bb);
        new llvm::StoreInst(arr,ptr,bb);
        args.push_back(a);
        new llvm::CallInst(mfn, args.begin(), args.end(), "", bb);
        call = new llvm::CallInst(ir.mainFunc,a,"ret",bb);
    }
    else
    {
        // main with no arguments
        call = new llvm::CallInst(ir.mainFunc,"ret",bb);
    }
    call->setCallingConv(ir.mainFunc->getCallingConv());

    // call static dtors
    fn = LLVM_D_GetRuntimeFunction(ir.module,"_d_run_module_dtors");
    new llvm::CallInst(fn,"",bb);

    // return
    new llvm::ReturnInst(call,bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoCallClassDtors(TypeClass* tc, llvm::Value* instance)
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

void DtoInitClass(TypeClass* tc, llvm::Value* dst)
{
    assert(gIR);

    assert(tc->llvmType);
    uint64_t size_t_size = gTargetData->getTypeSize(DtoSize_t());
    uint64_t n = gTargetData->getTypeSize(tc->llvmType) - size_t_size;

    // set vtable field
    llvm::Value* vtblvar = DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
    assert(tc->sym->llvmVtbl);
    new llvm::StoreInst(tc->sym->llvmVtbl, vtblvar, gIR->scopebb());

    // copy the static initializer
    if (n > 0) {
        assert(tc->llvmInit);
        assert(dst->getType() == tc->llvmInit->getType());

        llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

        llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
        dstarr = DtoGEPi(dstarr,size_t_size,"tmp",gIR->scopebb());

        llvm::Value* srcarr = new llvm::BitCastInst(tc->llvmInit,arrty,"tmp",gIR->scopebb());
        srcarr = DtoGEPi(srcarr,size_t_size,"tmp",gIR->scopebb());

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

llvm::Constant* DtoConstInitializer(Type* type, Initializer* init)
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
        _init = DtoConstStructInitializer(si);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        Logger::println("const array initializer");
        _init = DtoConstArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        Logger::println("const void initializer");
        const llvm::Type* ty = DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported const initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoInitializer(Initializer* init)
{
    if (ExpInitializer* ex = init->isExpInitializer())
    {
        Logger::println("expression initializer");
        assert(ex->exp);
        return ex->exp->toElem(gIR);
    }
    else if (init->isVoidInitializer())
    {
        // do nothing
    }
    else {
        Logger::println("unsupported initializer: %s", init->toChars());
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = i0;
    v[1] = i1;
    //Logger::cout() << "DtoGEP: " << *ptr << '\n';
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb)
{
    size_t n = src.size();
    std::vector<llvm::Value*> dst(n, NULL);
    //std::ostream& ostr = Logger::cout();
    //ostr << "indices for '" << *ptr << "':";
    for (size_t i=0; i<n; ++i)
    {
        //ostr << ' ' << i;
        dst[i] = llvm::ConstantInt::get(llvm::Type::Int32Ty, src[i], false);
    }
    //ostr << '\n';*/
    return new llvm::GetElementPtrInst(ptr, dst.begin(), dst.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i, const std::string& var, llvm::BasicBlock* bb)
{
    return new llvm::GetElementPtrInst(ptr, llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb)
{
    std::vector<llvm::Value*> v(2);
    v[0] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i0, false);
    v[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i1, false);
    return new llvm::GetElementPtrInst(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)DtoDType(fdecl->type);
    const llvm::FunctionType* fty = DtoVaFunctionType(fdecl);
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

    llvm::Function* func = llvm::dyn_cast<llvm::Function>(fn);
    assert(func);
    assert(func->isIntrinsic());
    fdecl->llvmValue = func;
    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* DtoDeclareFunction(FuncDeclaration* fdecl)
{
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        return DtoDeclareVaFunction(fdecl);
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
    TypeFunction* f = (TypeFunction*)DtoDType(fdecl->type);
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
    const llvm::FunctionType* functype = (f->llvmType == 0) ? DtoFunctionType(fdecl) : llvm::cast<llvm::FunctionType>(f->llvmType);

    // make the function
    llvm::Function* func = gIR->module->getFunction(mangled_name);
    if (func == 0) {
        func = new llvm::Function(functype,DtoLinkage(fdecl->protection, fdecl->storage_class),mangled_name,gIR->module);
    }

    if (fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(DtoCallingConv(f->linkage));

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
        //Logger::println("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
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

    Logger::cout() << "func decl: " << *func << '\n';

    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoRealloc(llvm::Value* ptr, const llvm::Type* ty)
{
    /*size_t sz = gTargetData->getTypeSize(ty);
    llvm::ConstantInt* n = llvm::ConstantInt::get(DtoSize_t(), sz, false);
    if (ptr == 0) {
        llvm::PointerType* i8pty = llvm::PointerType::get(llvm::Type::Int8Ty);
        ptr = llvm::ConstantPointerNull::get(i8pty);
    }
    return DtoRealloc(ptr, n);*/
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoRealloc(llvm::Value* ptr, llvm::Value* n)
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

void DtoAssert(llvm::Value* cond, Loc* loc, DValue* msg)
{
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_assert");
    const llvm::FunctionType* fnt = fn->getFunctionType();

    std::vector<llvm::Value*> llargs;
    llargs.resize(3);
    llargs[0] = cond ? DtoBoolean(cond) : llvm::ConstantInt::getFalse();
    llargs[1] = DtoConstUint(loc->linnum);
    llargs[2] = msg ? msg->getRVal() : llvm::Constant::getNullValue(fnt->getParamType(2));

    assert(fn);
    llvm::CallInst* call = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
    call->setCallingConv(llvm::CallingConv::C);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoArgument(const llvm::Type* paramtype, Argument* fnarg, Expression* argexp)
{
    llvm::Value* retval = 0;

    bool haslvals = !gIR->exps.empty();
    if (haslvals)
        gIR->exps.push_back(IRExp(NULL,NULL,NULL));

    DValue* arg = argexp->toElem(gIR);

    if (haslvals)
        gIR->exps.pop_back();

    if (arg->inPlace()) {
        retval = arg->getRVal();
        return retval;
    }

    Type* realtype = DtoDType(argexp->type);
    TY argty = realtype->ty;
    if (DtoIsPassedByRef(realtype)) {
        if (!fnarg || !fnarg->llvmCopy) {
            if (DSliceValue* sv = arg->isSlice()) {
                retval = new llvm::AllocaInst(DtoType(realtype), "tmpparam", gIR->topallocapoint());
                DtoSetArray(retval, DtoArrayLen(sv), DtoArrayPtr(sv));
            }
            else {
                retval = arg->getRVal();
            }
        }
        else {
            llvm::Value* allocaInst = 0;
            llvm::BasicBlock* entryblock = &gIR->topfunc()->front();

            const llvm::Type* realtypell = DtoType(realtype);
            const llvm::PointerType* pty = llvm::PointerType::get(realtypell);
            if (argty == Tstruct) {
                allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                DValue* dst = new DVarValue(realtype, allocaInst, true);
                DtoAssign(dst,arg);
                delete dst;
            }
            else if (argty == Tdelegate) {
                allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                DValue* dst = new DVarValue(realtype, allocaInst, true);
                DtoAssign(dst,arg);
                delete dst;
            }
            else if (argty == Tarray) {
                if (arg->isSlice()) {
                    allocaInst = new llvm::AllocaInst(realtypell, "tmpparam", gIR->topallocapoint());
                }
                else {
                    allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", gIR->topallocapoint());
                }
            }
            else
            assert(0);

            DValue* dst = new DVarValue(realtype, allocaInst, true);
            DtoAssign(dst,arg);
            delete dst;

            retval = allocaInst;
        }
    }
    else if (!fnarg || fnarg->llvmCopy) {
        Logger::println("regular arg");
        if (DSliceValue* sl = arg->isSlice()) {
            if (sl->ptr) Logger::cout() << "ptr = " << *sl->ptr << '\n';
            if (sl->len) Logger::cout() << "len = " << *sl->len << '\n';
            assert(0);
        }
        else {
            retval = arg->getRVal();
        }
    }
    else {
        Logger::println("as ptr arg");
        retval = arg->getLVal();
        if (paramtype && retval->getType() != paramtype)
        {
            assert(0);
            /*assert(retval->getType() == paramtype->getContainedType(0));
            new llvm::StoreInst(retval, arg->getLVal(), gIR->scopebb());
            retval = arg->getLVal();*/
        }
    }

    if (fnarg && paramtype && retval->getType() != paramtype) {
        // this is unfortunately needed with the way SymOffExp is overused
        // and static arrays can end up being a pointer to their element type
        if (arg->isField()) {
            retval = gIR->ir->CreateBitCast(retval, paramtype, "tmp");
        }
        else {
            Logger::cout() << "got '" << *retval->getType() << "' expected '" << *paramtype << "'\n";
            assert(0 && "parameter type that was actually passed is invalid");
        }
    }

    delete arg;

    return retval;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoNestedVariable(VarDeclaration* vd)
{
    FuncDeclaration* fd = vd->toParent()->isFuncDeclaration();
    assert(fd != NULL);

    IRFunction* fcur = &gIR->func();
    FuncDeclaration* f = fcur->decl;

    // on this stack
    if (fd == f) {
        llvm::Value* v = DtoGEPi(vd->llvmValue,0,unsigned(vd->llvmNestedIndex),"tmp");
        if (vd->isParameter() && (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type))) {
            Logger::cout() << "1267 loading: " << *v << '\n';
            v = gIR->ir->CreateLoad(v,"tmp");
        }
        return v;
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
            llvm::Value* v = DtoGEPi(ptr,0,vd->llvmNestedIndex,"tmp");
            if (vd->isParameter() && (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type))) {
                Logger::cout() << "1291 loading: " << *v << '\n';
                v = gIR->ir->CreateLoad(v,"tmp");
            }
            return v;
        }
        else {
            ptr = DtoGEPi(ptr,0,0,"tmp");
            ptr = gIR->ir->CreateLoad(ptr,"tmp");
        }
        f = f->toParent()->isFuncDeclaration();
    }

    assert(0 && "nested var not found");
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAssign(DValue* lhs, DValue* rhs)
{
    Logger::cout() << "DtoAssign(...);\n";
    Type* t = DtoDType(lhs->getType());
    Type* t2 = DtoDType(rhs->getType());

    if (t->ty == Tstruct) {
        if (t2 != t) {
            // TODO: fix this, use 'rhs' for something
            DtoStructZeroInit(lhs->getLVal());
        }
        else if (!rhs->inPlace()) {
            DtoStructCopy(lhs->getLVal(),rhs->getRVal());
        }
    }
    else if (t->ty == Tarray) {
        // lhs is slice
        if (DSliceValue* s = lhs->isSlice()) {
            if (DSliceValue* s2 = rhs->isSlice()) {
                DtoArrayCopy(s, s2);
            }
            else if (t->next == t2) {
                if (s->len)
                    DtoArrayInit(s->ptr, s->len, rhs->getRVal());
                else
                    DtoArrayInit(s->ptr, rhs->getRVal());
            }
            else
            assert(rhs->inPlace());
        }
        // rhs is slice
        else if (DSliceValue* s = rhs->isSlice()) {
            DtoSetArray(lhs->getLVal(),s->len,s->ptr);
        }
        // null
        else if (rhs->isNull()) {
            DtoNullArray(lhs->getLVal());
        }
        // reference assignment
        else {
            DtoArrayAssign(lhs->getLVal(), rhs->getRVal());
        }
    }
    else if (t->ty == Tsarray) {
        DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
    }
    else if (t->ty == Tdelegate) {
        if (rhs->isNull())
            DtoNullDelegate(lhs->getLVal());
        else if (!rhs->inPlace())
            DtoDelegateCopy(lhs->getLVal(), rhs->getRVal());
    }
    else if (t->ty == Tclass) {
        assert(t2->ty == Tclass);
        // assignment to this in constructor special case
        if (lhs->isThis()) {
            llvm::Value* tmp = rhs->getRVal();
            FuncDeclaration* fdecl = gIR->func().decl;
            // respecify the this param
            if (!llvm::isa<llvm::AllocaInst>(fdecl->llvmThisVar))
                fdecl->llvmThisVar = new llvm::AllocaInst(tmp->getType(), "newthis", gIR->topallocapoint());
            DtoStore(tmp, fdecl->llvmThisVar);
        }
        // regular class ref -> class ref assignment
        else {
            DtoStore(rhs->getRVal(), lhs->getLVal());
        }
    }
    else {
        llvm::Value* r = rhs->getRVal();
        llvm::Value* l = lhs->getLVal();
        Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        const llvm::Type* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            r = DtoBitCast(r, lit);
            Logger::cout() << "really assign\nlhs: " << *l << "rhs: " << *r << '\n';
        }
        gIR->ir->CreateStore(r, l);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
DValue* DtoCastInt(DValue* val, Type* _to)
{
    const llvm::Type* tolltype = DtoType(_to);

    Type* to = DtoDType(_to);
    Type* from = DtoDType(val->getType());
    assert(from->isintegral());

    size_t fromsz = from->size();
    size_t tosz = to->size();

    llvm::Value* rval;

    if (to->isintegral()) {
        if (fromsz < tosz) {
            Logger::cout() << "cast to: " << *tolltype << '\n';
            if (from->isunsigned() || from->ty == Tbool) {
                rval = new llvm::ZExtInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
            } else {
                rval = new llvm::SExtInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
            }
        }
        else if (fromsz > tosz) {
            rval = new llvm::TruncInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::BitCastInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
    }
    else if (to->isfloating()) {
        if (from->isunsigned()) {
            rval = new llvm::UIToFPInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::SIToFPInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
    }
    else if (to->ty == Tpointer) {
        rval = gIR->ir->CreateIntToPtr(val->getRVal(), tolltype, "tmp");
    }
    else {
        assert(0 && "bad int cast");
    }

    return new DImValue(_to, rval);
}

DValue* DtoCastPtr(DValue* val, Type* to)
{
    const llvm::Type* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(val->getType());
    assert(fromtype->ty == Tpointer);

    llvm::Value* rval;

    if (totype->ty == Tpointer || totype->ty == Tclass) {
        llvm::Value* src = val->getRVal();
        Logger::cout() << "src: " << *src << "to type: " << *tolltype << '\n';
        rval = new llvm::BitCastInst(src, tolltype, "tmp", gIR->scopebb());
    }
    else if (totype->isintegral()) {
        rval = new llvm::PtrToIntInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
    }
    else {
        assert(0);
    }

    return new DImValue(to, rval);
}

DValue* DtoCastFloat(DValue* val, Type* to)
{
    const llvm::Type* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(val->getType());
    assert(fromtype->isfloating());

    size_t fromsz = fromtype->size();
    size_t tosz = totype->size();

    llvm::Value* rval;

    if (totype->isfloating()) {
        if ((fromtype->ty == Tfloat80 || fromtype->ty == Tfloat64) && (totype->ty == Tfloat80 || totype->ty == Tfloat64)) {
            rval = val->getRVal();
        }
        else if (fromsz < tosz) {
            rval = new llvm::FPExtInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else if (fromsz > tosz) {
            rval = new llvm::FPTruncInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            assert(0 && "bad float cast");
        }
    }
    else if (totype->isintegral()) {
        if (totype->isunsigned()) {
            rval = new llvm::FPToUIInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::FPToSIInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
        }
    }
    else {
        assert(0 && "bad float cast");
    }

    return new DImValue(to, rval);
}

DValue* DtoCastClass(DValue* val, Type* _to)
{
    const llvm::Type* tolltype = DtoType(_to);
    Type* to = DtoDType(_to);
    assert(to->ty == Tclass || to->ty == Tpointer);
    llvm::Value* rval = new llvm::BitCastInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
    return new DImValue(_to, rval);
}

DValue* DtoCastArray(DValue* u, Type* to)
{
    const llvm::Type* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(u->getType());
    assert(fromtype->ty == Tarray || fromtype->ty == Tsarray);

    llvm::Value* rval;
    llvm::Value* rval2;
    bool isslice = false;

    Logger::cout() << "from array or sarray" << '\n';
    if (totype->ty == Tpointer) {
        Logger::cout() << "to pointer" << '\n';
        assert(fromtype->next == totype->next || totype->next->ty == Tvoid);
        llvm::Value* ptr = DtoGEPi(u->getRVal(),0,1,"tmp",gIR->scopebb());
        rval = new llvm::LoadInst(ptr, "tmp", gIR->scopebb());
        if (fromtype->next != totype->next)
            rval = gIR->ir->CreateBitCast(rval, llvm::PointerType::get(llvm::Type::Int8Ty), "tmp");
    }
    else if (totype->ty == Tarray) {
        Logger::cout() << "to array" << '\n';
        const llvm::Type* ptrty = DtoType(totype->next);
        if (ptrty == llvm::Type::VoidTy)
            ptrty = llvm::Type::Int8Ty;
        ptrty = llvm::PointerType::get(ptrty);

        const llvm::Type* ety = DtoType(fromtype->next);
        if (ety == llvm::Type::VoidTy)
            ety = llvm::Type::Int8Ty;

        if (DSliceValue* usl = u->isSlice()) {
            Logger::println("from slice");
            rval = new llvm::BitCastInst(usl->ptr, ptrty, "tmp", gIR->scopebb());
            if (fromtype->next->size() == totype->next->size())
                rval2 = usl->len;
            else
                rval2 = DtoArrayCastLength(usl->len, ety, ptrty->getContainedType(0));
        }
        else {
            llvm::Value* uval = u->getRVal();
            if (fromtype->ty == Tsarray) {
                Logger::cout() << "uvalTy = " << *uval->getType() << '\n';
                assert(isaPointer(uval->getType()));
                const llvm::ArrayType* arrty = isaArray(uval->getType()->getContainedType(0));
                rval2 = llvm::ConstantInt::get(DtoSize_t(), arrty->getNumElements(), false);
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));
                rval = new llvm::BitCastInst(uval, ptrty, "tmp", gIR->scopebb());
            }
            else {
                llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
                llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
                rval2 = DtoGEP(uval,zero,zero,"tmp",gIR->scopebb());
                rval2 = new llvm::LoadInst(rval2, "tmp", gIR->scopebb());
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));

                rval = DtoGEP(uval,zero,one,"tmp",gIR->scopebb());
                rval = new llvm::LoadInst(rval, "tmp", gIR->scopebb());
                //Logger::cout() << *e->mem->getType() << '|' << *ptrty << '\n';
                rval = new llvm::BitCastInst(rval, ptrty, "tmp", gIR->scopebb());
            }
        }
        isslice = true;
    }
    else if (totype->ty == Tsarray) {
        Logger::cout() << "to sarray" << '\n';
        assert(0);
    }
    else {
        assert(0);
    }

    if (isslice) {
        Logger::println("isslice");
        return new DSliceValue(to, rval2, rval);
    }

    return new DImValue(to, rval);
}

DValue* DtoCast(DValue* val, Type* to)
{
    Type* fromtype = DtoDType(val->getType());
    if (fromtype->isintegral()) {
        return DtoCastInt(val, to);
    }
    else if (fromtype->isfloating()) {
        return DtoCastFloat(val, to);
    }
    else if (fromtype->ty == Tclass) {
        return DtoCastClass(val, to);
    }
    else if (fromtype->ty == Tarray || fromtype->ty == Tsarray) {
        return DtoCastArray(val, to);
    }
    else if (fromtype->ty == Tpointer) {
        return DtoCastPtr(val, to);
    }
    else {
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::ConstantInt* DtoConstSize_t(size_t i)
{
    return llvm::ConstantInt::get(DtoSize_t(), i, false);
}
llvm::ConstantInt* DtoConstUint(unsigned i)
{
    return llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false);
}
llvm::ConstantInt* DtoConstInt(int i)
{
    return llvm::ConstantInt::get(llvm::Type::Int32Ty, i, true);
}
llvm::Constant* DtoConstBool(bool b)
{
    return llvm::ConstantInt::get(llvm::Type::Int1Ty, b, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* DtoConstString(const char* str)
{
    std::string s(str);
    llvm::Constant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, "stringliteral", gIR->module);
    llvm::Constant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    return DtoConstSlice(
        DtoConstSize_t(s.length()),
        llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2)
    );
}
llvm::Constant* DtoConstStringPtr(const char* str, const char* section)
{
    std::string s(str);
    llvm::Constant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, "stringliteral", gIR->module);
    if (section) gvar->setSection(section);
    llvm::Constant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    return llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes)
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

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoLoad(llvm::Value* src)
{
    return gIR->ir->CreateLoad(src,"tmp");
}

void DtoStore(llvm::Value* src, llvm::Value* dst)
{
    gIR->ir->CreateStore(src,dst);
}

bool DtoCanLoad(llvm::Value* ptr)
{
    if (isaPointer(ptr->getType())) {
        return ptr->getType()->getContainedType(0)->isFirstClassType();
    }
    return false;
}

llvm::Value* DtoBitCast(llvm::Value* v, const llvm::Type* t)
{
    if (v->getType() == t)
        return v;
    return gIR->ir->CreateBitCast(v, t, "tmp");
}

const llvm::PointerType* isaPointer(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::PointerType>(v->getType());
}

const llvm::PointerType* isaPointer(const llvm::Type* t)
{
    return llvm::dyn_cast<llvm::PointerType>(t);
}

const llvm::ArrayType* isaArray(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::ArrayType>(v->getType());
}

const llvm::ArrayType* isaArray(const llvm::Type* t)
{
    return llvm::dyn_cast<llvm::ArrayType>(t);
}

const llvm::StructType* isaStruct(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::StructType>(v->getType());
}

const llvm::StructType* isaStruct(const llvm::Type* t)
{
    return llvm::dyn_cast<llvm::StructType>(t);
}

llvm::Constant* isaConstant(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::Constant>(v);
}

llvm::ConstantInt* isaConstantInt(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::ConstantInt>(v);
}

llvm::Argument* isaArgument(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::Argument>(v);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool DtoIsTemplateInstance(Dsymbol* s)
{
    if (!s) return false;
    if (s->isTemplateInstance() && !s->isTemplateMixin())
        return true;
    else if (s->parent)
        return DtoIsTemplateInstance(s->parent);
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoLazyStaticInit(bool istempl, llvm::Value* gvar, Initializer* init, Type* t)
{
    // create a flag to make sure initialization only happens once
    llvm::GlobalValue::LinkageTypes gflaglink = istempl ? llvm::GlobalValue::WeakLinkage : llvm::GlobalValue::InternalLinkage;
    std::string gflagname(gvar->getName());
    gflagname.append("__initflag");
    llvm::GlobalVariable* gflag = new llvm::GlobalVariable(llvm::Type::Int1Ty,false,gflaglink,DtoConstBool(false),gflagname,gIR->module);

    // check flag and do init if not already done
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* initbb = new llvm::BasicBlock("ifnotinit",gIR->topfunc(),oldend);
    llvm::BasicBlock* endinitbb = new llvm::BasicBlock("ifnotinitend",gIR->topfunc(),oldend);
    llvm::Value* cond = gIR->ir->CreateICmpEQ(gIR->ir->CreateLoad(gflag,"tmp"),DtoConstBool(false));
    gIR->ir->CreateCondBr(cond, initbb, endinitbb);
    gIR->scope() = IRScope(initbb,endinitbb);
    DValue* ie = DtoInitializer(init);
    if (!ie->inPlace()) {
        DValue* dst = new DVarValue(t, gvar, true);
        DtoAssign(dst, ie);
    }
    gIR->ir->CreateStore(DtoConstBool(true), gflag);
    gIR->ir->CreateBr(endinitbb);
    gIR->scope() = IRScope(endinitbb,oldend);
}
