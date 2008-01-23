#include <iostream>

#include "gen/llvm.h"

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
            return (const llvm::Type*)getPtrToType(llvm::Type::Int8Ty);
        else
            return (const llvm::Type*)getPtrToType(DtoType(t->next));
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
                IrStruct* found = 0;
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
        return ts->sym->irStruct->recty.get();//t->llvmType->get();
    }

    case Tclass:    {
        /*if (!t->llvmType || *t->llvmType == NULL) {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IrStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == (*i)->type)
                    {
                        return getPtrToType((*i)->recty.get());
                    }
                }
            }
            Logger::println("no type found");
        }*/

        TypeClass* tc = (TypeClass*)t;
        assert(tc->sym);
        DtoResolveDsymbol(tc->sym);
        return getPtrToType(tc->sym->irStruct->recty.get());//t->llvmType->get());
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

    // associative arrays
    case Taarray:
    {
        TypeAArray* taa = (TypeAArray*)t;
        std::vector<const llvm::Type*> types;
        types.push_back(DtoType(taa->key));
        types.push_back(DtoType(taa->next));
        return getPtrToType(llvm::StructType::get(types));
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
    const llvm::Type* i8ptr = getPtrToType(llvm::Type::Int8Ty);
    const llvm::Type* func = DtoFunctionType(t->next, i8ptr);
    const llvm::Type* funcptr = getPtrToType(func);

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
    const llvm::Type* int8ptrty = (const llvm::Type*)getPtrToType(llvm::Type::Int8Ty);
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
    return llvm::cast<llvm::Function>(gIR->module->getOrInsertFunction(name, functype));
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

    const llvm::Type* i8p_ty = getPtrToType(llvm::Type::Int8Ty);

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

    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);

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
        if (stc & STCextern)
            return llvm::GlobalValue::ExternalLinkage;
        else
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
        llvm::Value* zero = llvm::Constant::getNullValue(t);
        return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, val, zero, "tmp", gIR->scopebb());
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
            assert(ts->sym->irStruct->constInit);
            _init = ts->sym->irStruct->constInit;
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
        llvm::PointerType* i8pty = getPtrToType(llvm::Type::Int8Ty);
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

    const llvm::PointerType* i8pty = getPtrToType(llvm::Type::Int8Ty);
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

void DtoAssert(Loc* loc, DValue* msg)
{
    std::vector<llvm::Value*> args;
    llvm::Constant* c;

    // func
    const char* fname = msg ? "_d_assert_msg" : "_d_assert";

    // msg param
    if (msg) args.push_back(msg->getRVal());

    // file param
    c = DtoConstString(loc->filename);
    llvm::AllocaInst* alloc = new llvm::AllocaInst(c->getType(), "srcfile", gIR->topallocapoint());
    llvm::Value* ptr = DtoGEPi(alloc, 0,0, "tmp");
    DtoStore(c->getOperand(0), ptr);
    ptr = DtoGEPi(alloc, 0,1, "tmp");
    DtoStore(c->getOperand(1), ptr);
    args.push_back(alloc);

    // line param
    c = DtoConstUint(loc->linnum);
    args.push_back(c);

    // call
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);
    llvm::CallInst* call = new llvm::CallInst(fn, args.begin(), args.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

static const llvm::Type* get_next_frame_ptr_type(Dsymbol* sc)
{
    assert(sc->isFuncDeclaration() || sc->isClassDeclaration());
    Dsymbol* p = sc->toParent2();
    if (!p->isFuncDeclaration() && !p->isClassDeclaration())
        Logger::println("unexpected parent symbol found while resolving frame pointer - '%s' kind: '%s'", p->toChars(), p->kind());
    assert(p->isFuncDeclaration() || p->isClassDeclaration());
    if (FuncDeclaration* fd = p->isFuncDeclaration())
    {
        llvm::Value* v = fd->irFunc->nestedVar;
        assert(v);
        return v->getType();
    }
    else if (ClassDeclaration* cd = p->isClassDeclaration())
    {
        return DtoType(cd->type);
    }
    else
    {
        Logger::println("symbol: '%s' kind: '%s'", sc->toChars(), sc->kind());
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Value* get_frame_ptr_impl(FuncDeclaration* func, Dsymbol* sc, llvm::Value* v)
{
    LOG_SCOPE;
    if (sc == func)
    {
        return v;
    }
    else if (FuncDeclaration* fd = sc->isFuncDeclaration())
    {
        Logger::println("scope is function: %s", fd->toChars());

        if (fd->toParent2() == func)
        {
            if (!func->irFunc->nestedVar)
                return NULL;
            return DtoBitCast(v, func->irFunc->nestedVar->getType());
        }

        v = DtoBitCast(v, get_next_frame_ptr_type(fd));
        Logger::cout() << "v = " << *v << '\n';

        if (fd->toParent2()->isFuncDeclaration())
        {
            v = DtoGEPi(v, 0,0, "tmp");
            v = DtoLoad(v);
        }
        else if (ClassDeclaration* cd = fd->toParent2()->isClassDeclaration())
        {
            size_t idx = 2;
            idx += cd->irStruct->interfaces.size();
            v = DtoGEPi(v,0,idx,"tmp");
            v = DtoLoad(v);
        }
        else
        {
            assert(0);
        }
        return get_frame_ptr_impl(func, fd->toParent2(), v);
    }
    else if (ClassDeclaration* cd = sc->isClassDeclaration())
    {
        Logger::println("scope is class: %s", cd->toChars());
        /*size_t idx = 2;
        idx += cd->llvmIrStruct->interfaces.size();
        v = DtoGEPi(v,0,idx,"tmp");
        Logger::cout() << "gep = " << *v << '\n';
        v = DtoLoad(v);*/
        return get_frame_ptr_impl(func, cd->toParent2(), v);
    }
    else
    {
        Logger::println("symbol: '%s'", sc->toPrettyChars());
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Value* get_frame_ptr(FuncDeclaration* func)
{
    Logger::println("Resolving context pointer for nested function: '%s'", func->toPrettyChars());
    LOG_SCOPE;
    IrFunction* irfunc = gIR->func();

    // in the right scope already
    if (func == irfunc->decl)
        return irfunc->decl->irFunc->nestedVar;

    // use the 'this' pointer
    llvm::Value* ptr = irfunc->decl->irFunc->thisVar;
    assert(ptr);

    // return the fully resolved frame pointer
    ptr = get_frame_ptr_impl(func, irfunc->decl, ptr);
    if (ptr) Logger::cout() << "Found context!" << *ptr;
    else Logger::cout() << "NULL context!\n";

    return ptr;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoNestedContext(FuncDeclaration* func)
{
    // resolve frame ptr
    llvm::Value* ptr = get_frame_ptr(func);
    Logger::cout() << "Nested context ptr = ";
    if (ptr) Logger::cout() << *ptr;
    else Logger::cout() << "NULL";
    Logger::cout() << '\n';
    return ptr;
}

//////////////////////////////////////////////////////////////////////////////////////////

static void print_frame_worker(VarDeclaration* vd, Dsymbol* par)
{
    if (vd->toParent2() == par)
    {
        Logger::println("found: '%s' kind: '%s'", par->toChars(), par->kind());
        return;
    }

    Logger::println("diving into: '%s' kind: '%s'", par->toChars(), par->kind());
    LOG_SCOPE;
    print_frame_worker(vd, par->toParent2());
}

//////////////////////////////////////////////////////////////////////////////////////////

static void print_nested_frame_list(VarDeclaration* vd, Dsymbol* par)
{
    Logger::println("Frame pointer list for nested var: '%s'", vd->toPrettyChars());
    LOG_SCOPE;
    if (vd->toParent2() != par)
        print_frame_worker(vd, par);
    else
        Logger::println("Found at level 0");
    Logger::println("Done");
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoNestedVariable(VarDeclaration* vd)
{
    // log the frame list
    IrFunction* irfunc = gIR->func();
    if (Logger::enabled())
        print_nested_frame_list(vd, irfunc->decl);

    // resolve frame ptr
    FuncDeclaration* func = vd->toParent2()->isFuncDeclaration();
    assert(func);
    llvm::Value* ptr = DtoNestedContext(func);
    assert(ptr && "nested var, but no context");

    // we must cast here to be sure. nested classes just have a void*
    ptr = DtoBitCast(ptr, func->irFunc->nestedVar->getType());

    // index nested var and load (if necessary)
    llvm::Value* v = DtoGEPi(ptr, 0, vd->irLocal->nestedIndex, "tmp");
    // references must be loaded, for normal variables this IS already the variable storage!!!
    if (vd->isParameter() && (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type)))
        v = DtoLoad(v);

    // log and return
    Logger::cout() << "Nested var ptr = " << *v << '\n';
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAssign(DValue* lhs, DValue* rhs)
{
    Logger::cout() << "DtoAssign(...);\n";
    LOG_SCOPE;

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
                DtoArrayCopySlices(s, s2);
            }
            else if (t->next == t2) {
                if (s->len)
                    DtoArrayInit(s->ptr, s->len, rhs->getRVal());
                else
                    DtoArrayInit(s->ptr, rhs->getRVal());
            }
            else {
                DtoArrayCopyToSlice(s, rhs);
            }
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
        else if (!rhs->inPlace()) {
            llvm::Value* l = lhs->getLVal();
            llvm::Value* r = rhs->getRVal();
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
            DtoDelegateCopy(l, r);
        }
    }
    else if (t->ty == Tclass) {
        assert(t2->ty == Tclass);
        // assignment to this in constructor special case
        if (lhs->isThis()) {
            llvm::Value* tmp = rhs->getRVal();
            FuncDeclaration* fdecl = gIR->func()->decl;
            // respecify the this param
            if (!llvm::isa<llvm::AllocaInst>(fdecl->irFunc->thisVar))
                fdecl->irFunc->thisVar = new llvm::AllocaInst(tmp->getType(), "newthis", gIR->topallocapoint());
            DtoStore(tmp, fdecl->irFunc->thisVar);
        }
        // regular class ref -> class ref assignment
        else {
            DtoStore(rhs->getRVal(), lhs->getLVal());
        }
    }
    else if (t->iscomplex()) {
        assert(!lhs->isComplex());

        llvm::Value* dst;
        if (DLRValue* lr = lhs->isLRValue()) {
            dst = lr->getLVal();
            rhs = DtoCastComplex(rhs, lr->getLType());
        }
        else {
            dst = lhs->getRVal();
        }

        if (DComplexValue* cx = rhs->isComplex())
            DtoComplexSet(dst, cx->re, cx->im);
        else
            DtoComplexAssign(dst, rhs->getRVal());
    }
    else {
        llvm::Value* l = lhs->getLVal();
        llvm::Value* r = rhs->getRVal();
        Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        const llvm::Type* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            if (DLRValue* lr = lhs->isLRValue()) // handle lvalue cast assignments
                r = DtoCast(rhs, lr->getLType())->getRVal();
            else
                r = DtoCast(rhs, lhs->getType())->getRVal();
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

    llvm::Value* rval = val->getRVal();
    if (rval->getType() == tolltype) {
        return new DImValue(_to, rval);
    }

    if (to->isintegral()) {
        if (fromsz < tosz) {
            Logger::cout() << "cast to: " << *tolltype << '\n';
            if (from->isunsigned() || from->ty == Tbool) {
                rval = new llvm::ZExtInst(rval, tolltype, "tmp", gIR->scopebb());
            } else {
                rval = new llvm::SExtInst(rval, tolltype, "tmp", gIR->scopebb());
            }
        }
        else if (fromsz > tosz) {
            rval = new llvm::TruncInst(rval, tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::BitCastInst(rval, tolltype, "tmp", gIR->scopebb());
        }
    }
    else if (to->isfloating()) {
        if (from->isunsigned()) {
            rval = new llvm::UIToFPInst(rval, tolltype, "tmp", gIR->scopebb());
        }
        else {
            rval = new llvm::SIToFPInst(rval, tolltype, "tmp", gIR->scopebb());
        }
    }
    else if (to->ty == Tpointer) {
        Logger::cout() << "cast pointer: " << *tolltype << '\n';
        rval = gIR->ir->CreateIntToPtr(rval, tolltype, "tmp");
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
        Logger::println("invalid cast from '%s' to '%s'", val->getType()->toChars(), to->toChars());
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
    Type* vty = val->getType();
    if (to->iscomplex()) {
        if (vty->size() == to->size())
            return val;

        llvm::Value *re, *im;
        DtoGetComplexParts(val, re, im);
        const llvm::Type* toty = DtoComplexBaseType(to);

        if (to->size() < vty->size()) {
            re = gIR->ir->CreateFPTrunc(re, toty, "tmp");
            im = gIR->ir->CreateFPTrunc(im, toty, "tmp");
        }
        else if (to->size() > vty->size()) {
            re = gIR->ir->CreateFPExt(re, toty, "tmp");
            im = gIR->ir->CreateFPExt(im, toty, "tmp");
        }
        else {
            return val;
        }

        if (val->isComplex())
            return new DComplexValue(_to, re, im);

        // unfortunately at this point, the cast value can show up as the lvalue for += and similar expressions.
        // so we need to give it storage, or fix the system that handles this stuff (DLRValue)
        llvm::Value* mem = new llvm::AllocaInst(DtoType(_to), "castcomplextmp", gIR->topallocapoint());
        DtoComplexSet(mem, re, im);
        return new DLRValue(val->getType(), val->getRVal(), _to, mem);
    }
    else if (to->isimaginary()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->im);
        llvm::Value* v = val->getRVal();
        DImValue* im = new DImValue(to, DtoLoad(DtoGEPi(v,0,1,"tmp")));
        return DtoCastFloat(im, to);
    }
    else if (to->isfloating()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->re);
        llvm::Value* v = val->getRVal();
        DImValue* re = new DImValue(to, DtoLoad(DtoGEPi(v,0,0,"tmp")));
        return DtoCastFloat(re, to);
    }
    else
    assert(0);
}

DValue* DtoCast(DValue* val, Type* to)
{
    Type* fromtype = DtoDType(val->getType());
    Logger::println("Casting from '%s' to '%s'", fromtype->toChars(), to->toChars());
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
        return llvm::ConstantFP::get(llvm::Type::FloatTy, llvm::APFloat(float(value)));
    else if (ty == Tfloat64 || ty == Timaginary64 || ty == Tfloat80 || ty == Timaginary80)
        return llvm::ConstantFP::get(llvm::Type::DoubleTy, llvm::APFloat(double(value)));
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
        getPtrToType(t)
    );
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(llvm::Value* dst, llvm::Value* nbytes)
{
    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);
    llvm::Value *dstarr;
    if (dst->getType() == arrty)
    {
        dstarr = dst;
    }
    else
    {
        dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    }

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemSet64() : LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = nbytes;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes)
{
    assert(dst->getType() == src->getType());

    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);
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

llvm::Value* DtoBitCast(llvm::Value* v, const llvm::Type* t, const char* name)
{
    if (v->getType() == t)
        return v;
    return gIR->ir->CreateBitCast(v, t, name ? name : "tmp");
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

const llvm::PointerType* getPtrToType(const llvm::Type* t)
{
    return llvm::PointerType::get(t, 0);
}

llvm::ConstantPointerNull* getNullPtr(const llvm::Type* t)
{
    const llvm::PointerType* pt = llvm::cast<llvm::PointerType>(t);
    return llvm::ConstantPointerNull::get(pt);
}

//////////////////////////////////////////////////////////////////////////////////////////

size_t getTypeBitSize(const llvm::Type* t)
{
    return gTargetData->getTypeSizeInBits(t);
}

size_t getTypeStoreSize(const llvm::Type* t)
{
    return gTargetData->getTypeStoreSize(t);
}

size_t getABITypeSize(const llvm::Type* t)
{
    return gTargetData->getABITypeSize(t);
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
            assert(ts->sym->irStruct->constInit);
            _init = ts->sym->irStruct->constInit;
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
    llvm::cast<llvm::OpaqueType>(vd->irGlobal->type.get())->refineAbstractTypeTo(_type);
    _type = vd->irGlobal->type.get();
    assert(!_type->isAbstract());

    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(vd->irGlobal->value);
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
void DtoEmptyAllLists()
{
    for(;;)
    {
        Dsymbol* dsym;
        if (!gIR->resolveList.empty()) {
            dsym = gIR->resolveList.front();
            gIR->resolveList.pop_front();
            DtoResolveDsymbol(dsym);
        }
        else if (!gIR->declareList.empty()) {
            dsym = gIR->declareList.front();
            gIR->declareList.pop_front();
            DtoDeclareDsymbol(dsym);
        }
        else if (!gIR->constInitList.empty()) {
            dsym = gIR->constInitList.front();
            gIR->constInitList.pop_front();
            DtoConstInitDsymbol(dsym);
        }
        else if (!gIR->defineList.empty()) {
            dsym = gIR->defineList.front();
            gIR->defineList.pop_front();
            DtoDefineDsymbol(dsym);
        }
        else {
            break;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceDeclareDsymbol(Dsymbol* dsym)
{
    if (dsym->llvmDeclared) return;
    Logger::println("DtoForceDeclareDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();

    DtoDeclareDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceConstInitDsymbol(Dsymbol* dsym)
{
    if (dsym->llvmInitialized) return;
    Logger::println("DtoForceConstInitDsymbol(%s)", dsym->toPrettyChars());
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
    Logger::println("DtoForceDefineDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();
    DtoEmptyDeclareList();
    DtoEmptyConstInitList();

    DtoDefineDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAnnotation(const char* str)
{
    std::string s("CODE: ");
    s.append(str);
    char* p = &s[0];
    while (*p)
    {
        if (*p == '"')
            *p = '\'';
        ++p;
    }
    // create a noop with the code as the result name!
    gIR->ir->CreateAnd(DtoConstSize_t(0),DtoConstSize_t(0),s.c_str());
}

const llvm::StructType* DtoInterfaceInfoType()
{
    static const llvm::StructType* t = NULL;
    if (t)
        return t;

    // build interface info type
    std::vector<const llvm::Type*> types;
    // ClassInfo classinfo
    ClassDeclaration* cd2 = ClassDeclaration::classinfo;
    DtoResolveClass(cd2);
    types.push_back(getPtrToType(cd2->type->llvmType->get()));
    // void*[] vtbl
    std::vector<const llvm::Type*> vtbltypes;
    vtbltypes.push_back(DtoSize_t());
    const llvm::Type* byteptrptrty = getPtrToType(getPtrToType(llvm::Type::Int8Ty));
    vtbltypes.push_back(byteptrptrty);
    types.push_back(llvm::StructType::get(vtbltypes));
    // int offset
    types.push_back(llvm::Type::Int32Ty);
    // create type
    t = llvm::StructType::get(types);

    return t;
}
