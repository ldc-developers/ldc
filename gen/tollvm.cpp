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
#include "gen/functions.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/typeinf.h"
#include "gen/complex.h"

bool DtoIsPassedByRef(Type* type)
{
    Type* typ = DtoDType(type);
    TY t = typ->ty;
    return (t == Tstruct || t == Tarray || t == Tdelegate || t == Tsarray || typ->iscomplex());
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
    case Tcomplex64:
    case Tcomplex80:
        return DtoComplexType(t);

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
        if (!t->llvmType || *t->llvmType == NULL) {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IRStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == (*i)->type)
                    {
                        return (*i)->recty.get();
                    }
                }
            }
        }

        TypeStruct* ts = (TypeStruct*)t;
        assert(ts->sym);
        DtoResolveDsymbol(ts->sym);
        return t->llvmType->get();
    }

    case Tclass:    {
        if (!t->llvmType || *t->llvmType == NULL) {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IRStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == (*i)->type)
                    {
                        return llvm::PointerType::get((*i)->recty.get());
                    }
                }
            }
        }

        TypeClass* tc = (TypeClass*)t;
        assert(tc->sym);
        DtoResolveDsymbol(tc->sym);
        return llvm::PointerType::get(t->llvmType->get());
    }

    // functions
    case Tfunction:
    {
        if (!t->llvmType || *t->llvmType == NULL) {
            return DtoFunctionType(t,NULL);
        }
        else {
            return t->llvmType->get();
        }
    }

    // delegates
    case Tdelegate:
    {
        if (!t->llvmType || *t->llvmType == NULL) {
            return DtoDelegateType(t);
        }
        else {
            return t->llvmType->get();
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

llvm::Constant* DtoConstFieldInitializer(Type* t, Initializer* init)
{
    Logger::println("DtoConstFieldInitializer");
    LOG_SCOPE;

    const llvm::Type* _type = DtoType(t);

    llvm::Constant* _init = DtoConstInitializer(t, init);
    assert(_init);
    if (_type != _init->getType())
    {
        Logger::cout() << "field init is: " << *_init << " type should be " << *_type << '\n';
        if (t->ty == Tsarray)
        {
            const llvm::ArrayType* arrty = isaArray(_type);
            uint64_t n = arrty->getNumElements();
            std::vector<llvm::Constant*> vals(n,_init);
            _init = llvm::ConstantArray::get(arrty, vals);
        }
        else if (t->ty == Tarray)
        {
            assert(isaStruct(_type));
            _init = llvm::ConstantAggregateZero::get(_type);
        }
        else if (t->ty == Tstruct)
        {
            const llvm::StructType* structty = isaStruct(_type);
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts);
            assert(ts->sym);
            assert(ts->sym->llvmInitZ);
            _init = ts->sym->llvmInitZ;
        }
        else if (t->ty == Tclass)
        {
            _init = llvm::Constant::getNullValue(_type);
        }
        else {
            Logger::println("failed for type %s", t->toChars());
            assert(0);
        }
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
    if (msg)
        llargs[2] = msg->getRVal();
    else {
        llvm::Constant* c = DtoConstSlice(DtoConstSize_t(0), DtoConstNullPtr(llvm::Type::Int8Ty));
        static llvm::AllocaInst* alloc = 0;
        if (!alloc || alloc->getParent()->getParent() != gIR->func()->func) {
            alloc = new llvm::AllocaInst(c->getType(), "assertnullparam", gIR->topallocapoint());
            DtoSetArrayToNull(alloc);
        }
        llargs[2] = alloc;
    }

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

    IRFunction* fcur = gIR->func();
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
            DtoSetArrayToNull(lhs->getLVal());
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
            FuncDeclaration* fdecl = gIR->func()->decl;
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
    else if (t->iscomplex()) {
        assert(!lhs->isComplex());
        if (DComplexValue* cx = rhs->isComplex()) {
            DtoComplexSet(lhs->getRVal(), cx->re, cx->im);
        }
        else {
            DtoComplexAssign(lhs->getRVal(), rhs->getRVal());
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
    if (val->getType() == to)
        return val;

    const llvm::Type* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(val->getType());
    assert(fromtype->isfloating());

    size_t fromsz = fromtype->size();
    size_t tosz = totype->size();

    llvm::Value* rval;

    if (totype->iscomplex()) {
        assert(0);
        //return new DImValue(to, DtoComplex(to, val));
    }
    else if (totype->isfloating()) {
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

DValue* DtoCastComplex(DValue* val, Type* _to)
{
    Type* to = DtoDType(_to);
    llvm::Value* v = val->getRVal();
    if (to->iscomplex()) {
        assert(0);
    }
    else if (to->isimaginary()) {
        DImValue* im = new DImValue(to, gIR->ir->CreateExtractElement(v, DtoConstUint(1), "im"));
        return DtoCastFloat(im, to);
    }
    else if (to->isfloating()) {
        DImValue* re = new DImValue(to, gIR->ir->CreateExtractElement(v, DtoConstUint(0), "re"));
        return DtoCastFloat(re, to);
    }
    else
    assert(0);
}

DValue* DtoCastClass(DValue* val, Type* _to)
{
    const llvm::Type* tolltype = DtoType(_to);
    Type* to = DtoDType(_to);
    assert(to->ty == Tclass || to->ty == Tpointer);
    llvm::Value* rval = new llvm::BitCastInst(val->getRVal(), tolltype, "tmp", gIR->scopebb());
    return new DImValue(_to, rval);
}

DValue* DtoCast(DValue* val, Type* to)
{
    Type* fromtype = DtoDType(val->getType());
    if (fromtype->isintegral()) {
        return DtoCastInt(val, to);
    }
    else if (fromtype->iscomplex()) {
        return DtoCastComplex(val, to);
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

llvm::ConstantFP* DtoConstFP(Type* t, long double value)
{
    TY ty = DtoDType(t)->ty;
    if (ty == Tfloat32 || ty == Timaginary32)
        return llvm::ConstantFP::get(llvm::Type::FloatTy, float(value));
    else if (ty == Tfloat64 || ty == Timaginary64 || ty == Tfloat80 || ty == Timaginary80)
        return llvm::ConstantFP::get(llvm::Type::DoubleTy, double(value));
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

llvm::Constant* DtoConstNullPtr(const llvm::Type* t)
{
    return llvm::ConstantPointerNull::get(
        llvm::PointerType::get(t)
    );
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

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoBitCast(llvm::Value* v, const llvm::Type* t)
{
    if (v->getType() == t)
        return v;
    return gIR->ir->CreateBitCast(v, t, "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

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

llvm::GlobalVariable* isaGlobalVar(llvm::Value* v)
{
    return llvm::dyn_cast<llvm::GlobalVariable>(v);
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

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoResolveStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoResolveClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoResolveFunction(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoResolveTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoResolveDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoDeclareStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoDeclareClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoDeclareFunction(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoDeclareTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoDeclareDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoConstInitStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoConstInitClass(cd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoConstInitTypeInfo(fd);
    }
    else if (VarDeclaration* vd = dsym->isVarDeclaration()) {
        DtoConstInitGlobal(vd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoConstInitDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineDsymbol(Dsymbol* dsym)
{
    if (StructDeclaration* sd = dsym->isStructDeclaration()) {
        DtoDefineStruct(sd);
    }
    else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
        DtoDefineClass(cd);
    }
    else if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
        DtoDefineFunc(fd);
    }
    else if (TypeInfoDeclaration* fd = dsym->isTypeInfoDeclaration()) {
        DtoDefineTypeInfo(fd);
    }
    else {
    error(dsym->loc, "unsupported dsymbol: %s", dsym->toChars());
    assert(0 && "unsupported dsymbol for DtoDefineDsymbol");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitGlobal(VarDeclaration* vd)
{
    if (vd->llvmInitialized) return;
    vd->llvmInitialized = gIR->dmodule;

    Logger::println("* DtoConstInitGlobal(%s)", vd->toChars());
    LOG_SCOPE;

    bool emitRTstaticInit = false;

    llvm::Constant* _init = 0;
    if (vd->parent && vd->parent->isFuncDeclaration() && vd->init && vd->init->isExpInitializer()) {
        _init = DtoConstInitializer(vd->type, NULL);
        emitRTstaticInit = true;
    }
    else {
        _init = DtoConstInitializer(vd->type, vd->init);
    }

    const llvm::Type* _type = DtoType(vd->type);
    Type* t = DtoDType(vd->type);

    //Logger::cout() << "initializer: " << *_init << '\n';
    if (_type != _init->getType()) {
        Logger::cout() << "got type '" << *_init->getType() << "' expected '" << *_type << "'\n";
        // zero initalizer
        if (_init->isNullValue())
            _init = llvm::Constant::getNullValue(_type);
        // pointer to global constant (struct.init)
        else if (llvm::isa<llvm::GlobalVariable>(_init))
        {
            assert(_init->getType()->getContainedType(0) == _type);
            llvm::GlobalVariable* gv = llvm::cast<llvm::GlobalVariable>(_init);
            assert(t->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts->sym->llvmInitZ);
            _init = ts->sym->llvmInitZ;
        }
        // array single value init
        else if (isaArray(_type))
        {
            _init = DtoConstStaticArray(_type, _init);
        }
        else {
            Logger::cout() << "Unexpected initializer type: " << *_type << '\n';
            //assert(0);
        }
    }

    bool istempl = false;
    if ((vd->storage_class & STCcomdat) || (vd->parent && DtoIsTemplateInstance(vd->parent))) {
        istempl = true;
    }

    if (_init && _init->getType() != _type)
        _type = _init->getType();
    llvm::cast<llvm::OpaqueType>(vd->llvmIRGlobal->type.get())->refineAbstractTypeTo(_type);
    _type = vd->llvmIRGlobal->type.get();
    assert(!_type->isAbstract());

    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(vd->llvmValue);
    if (!(vd->storage_class & STCextern) && (vd->getModule() == gIR->dmodule || istempl))
    {
        gvar->setInitializer(_init);
    }

    if (emitRTstaticInit)
        DtoLazyStaticInit(istempl, gvar, vd->init, t);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyResolveList()
{
    //Logger::println("DtoEmptyResolveList()");
    Dsymbol* dsym;
    while (!gIR->resolveList.empty()) {
        dsym = gIR->resolveList.front();
        gIR->resolveList.pop_front();
        DtoResolveDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyDeclareList()
{
    //Logger::println("DtoEmptyDeclareList()");
    Dsymbol* dsym;
    while (!gIR->declareList.empty()) {
        dsym = gIR->declareList.front();
        gIR->declareList.pop_front();
        DtoDeclareDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyConstInitList()
{
    //Logger::println("DtoEmptyConstInitList()");
    Dsymbol* dsym;
    while (!gIR->constInitList.empty()) {
        dsym = gIR->constInitList.front();
        gIR->constInitList.pop_front();
        DtoConstInitDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoEmptyDefineList()
{
    //Logger::println("DtoEmptyDefineList()");
    Dsymbol* dsym;
    while (!gIR->defineList.empty()) {
        dsym = gIR->defineList.front();
        gIR->defineList.pop_front();
        DtoDefineDsymbol(dsym);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceDeclareDsymbol(Dsymbol* dsym)
{
    if (dsym->llvmDeclared) return;
    Logger::println("DtoForceDeclareDsymbol(%s)", dsym->toChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();

    DtoDeclareDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceConstInitDsymbol(Dsymbol* dsym)
{
    if (dsym->llvmInitialized) return;
    Logger::println("DtoForceConstInitDsymbol(%s)", dsym->toChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();
    DtoEmptyDeclareList();

    DtoConstInitDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceDefineDsymbol(Dsymbol* dsym)
{
    if (dsym->llvmDefined) return;
    Logger::println("DtoForceDefineDsymbol(%s)", dsym->toChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();
    DtoEmptyDeclareList();
    DtoEmptyConstInitList();

    DtoDefineDsymbol(dsym);
}
