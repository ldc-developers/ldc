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

bool DtoIsReturnedInArg(Type* type)
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

const LLType* DtoType(Type* t)
{
    assert(t);
    switch (t->ty)
    {
    // integers
    case Tint8:
    case Tuns8:
    case Tchar:
        return (const LLType*)llvm::Type::Int8Ty;
    case Tint16:
    case Tuns16:
    case Twchar:
        return (const LLType*)llvm::Type::Int16Ty;
    case Tint32:
    case Tuns32:
    case Tdchar:
        return (const LLType*)llvm::Type::Int32Ty;
    case Tint64:
    case Tuns64:
        return (const LLType*)llvm::Type::Int64Ty;

    case Tbool:
        return (const LLType*)llvm::ConstantInt::getTrue()->getType();

    // floats
    case Tfloat32:
    case Timaginary32:
        return llvm::Type::FloatTy;
    case Tfloat64:
    case Timaginary64:
        return llvm::Type::DoubleTy;
    case Tfloat80:
    case Timaginary80:
        return (global.params.useFP80) ? llvm::Type::X86_FP80Ty : llvm::Type::DoubleTy;

    // complex
    case Tcomplex32:
    case Tcomplex64:
    case Tcomplex80:
        return DtoComplexType(t);

    // pointers
    case Tpointer: {
        assert(t->next);
        if (t->next->ty == Tvoid)
            return (const LLType*)getPtrToType(llvm::Type::Int8Ty);
        else
            return (const LLType*)getPtrToType(DtoType(t->next));
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
        if (!t->ir.type || *t->ir.type == NULL) {
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
        return ts->sym->ir.irStruct->recty.get(); // t->ir.type->get();
    }

    case Tclass:    {
        /*if (!t].type || *gIR->irType[t->ir.type == NULL) {
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
        std::vector<const LLType*> types;
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
    const LLType* i8ptr = getPtrToType(llvm::Type::Int8Ty);
    const LLType* func = DtoFunctionType(t->next, i8ptr);
    const LLType* funcptr = getPtrToType(func);

    std::vector<const LLType*> types;
    types.push_back(i8ptr);
    types.push_back(funcptr);
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
static llvm::Function* LLVM_DeclareMemIntrinsic(const char* name, int bits, bool set=false)
{
    assert(bits == 32 || bits == 64);
    const LLType* int8ty =    (const LLType*)llvm::Type::Int8Ty;
    const LLType* int32ty =   (const LLType*)llvm::Type::Int32Ty;
    const LLType* int64ty =   (const LLType*)llvm::Type::Int64Ty;
    const LLType* int8ptrty = (const LLType*)getPtrToType(llvm::Type::Int8Ty);
    const LLType* voidty =    (const LLType*)llvm::Type::VoidTy;

    assert(gIR);
    assert(gIR->module);

    // parameter types
    std::vector<const LLType*> pvec;
    pvec.push_back(int8ptrty);
    pvec.push_back(set?int8ty:int8ptrty);
    pvec.push_back(bits==32?int32ty:int64ty);
    pvec.push_back(int32ty);
    llvm::FunctionType* functype = llvm::FunctionType::get(voidty, pvec, false);
    return llvm::cast<llvm::Function>(gIR->module->getOrInsertFunction(name, functype));
}
*/

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memset.i32
llvm::Function* LLVM_DeclareMemSet32()
{
    return GET_INTRINSIC_DECL(memset_i32);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_DeclareMemSet64()
{
    return GET_INTRINSIC_DECL(memset_i64);
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i32
llvm::Function* LLVM_DeclareMemCpy32()
{
    return GET_INTRINSIC_DECL(memcpy_i32);
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i64
llvm::Function* LLVM_DeclareMemCpy64()
{
    return GET_INTRINSIC_DECL(memcpy_i64);
}

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

void DtoDelegateToNull(LLValue* v)
{
    LLSmallVector<LLValue*, 4> args;
    args.push_back(DtoBitCast(v, getVoidPtrType()));
    args.push_back(llvm::Constant::getNullValue(llvm::Type::Int8Ty));
    args.push_back(DtoConstInt(global.params.is64bit ? 16 : 8));
    args.push_back(DtoConstInt(0));
    gIR->ir->CreateCall(GET_INTRINSIC_DECL(memset_i32), args.begin(), args.end(), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDelegateCopy(LLValue* dst, LLValue* src)
{
    LLSmallVector<LLValue*, 4> args;
    args.push_back(DtoBitCast(dst,getVoidPtrType()));
    args.push_back(DtoBitCast(src,getVoidPtrType()));
    args.push_back(DtoConstInt(global.params.is64bit ? 16 : 8));
    args.push_back(DtoConstInt(0));
    gIR->ir->CreateCall(GET_INTRINSIC_DECL(memcpy_i32), args.begin(), args.end(), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoDelegateCompare(TOK op, LLValue* lhs, LLValue* rhs)
{
    Logger::println("Doing delegate compare");
    llvm::ICmpInst::Predicate pred = (op == TOKequal || op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    llvm::Value *b1, *b2;
    if (rhs == NULL)
    {
        LLValue* l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,0,"tmp"),"tmp");
        LLValue* r = llvm::Constant::getNullValue(l->getType());
        b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
        l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,1,"tmp"),"tmp");
        r = llvm::Constant::getNullValue(l->getType());
        b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    }
    else
    {
        LLValue* l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,0,"tmp"),"tmp");
        LLValue* r = gIR->ir->CreateLoad(DtoGEPi(rhs,0,0,"tmp"),"tmp");
        b1 = gIR->ir->CreateICmp(pred,l,r,"tmp");
        l = gIR->ir->CreateLoad(DtoGEPi(lhs,0,1,"tmp"),"tmp");
        r = gIR->ir->CreateLoad(DtoGEPi(rhs,0,1,"tmp"),"tmp");
        b2 = gIR->ir->CreateICmp(pred,l,r,"tmp");
    }
    LLValue* b = gIR->ir->CreateAnd(b1,b2,"tmp");
    if (op == TOKnotequal || op == TOKnotidentity)
        return gIR->ir->CreateNot(b,"tmp");
    return b;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym)
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

// llvm linkage types
/*      ExternalLinkage = 0, LinkOnceLinkage, WeakLinkage, AppendingLinkage,
  InternalLinkage, DLLImportLinkage, DLLExportLinkage, ExternalWeakLinkage,
  GhostLinkage */
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

LLValue* DtoBoolean(LLValue* val)
{
    const LLType* t = val->getType();
    if (t->isInteger())
    {
        if (t == llvm::Type::Int1Ty)
            return val;
        else {
            LLValue* zero = llvm::ConstantInt::get(t, 0, false);
            return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, val, zero, "tmp", gIR->scopebb());
        }
    }
    else if (isaPointer(t)) {
        LLValue* zero = llvm::Constant::getNullValue(t);
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

const LLType* DtoSize_t()
{
    if (global.params.is64bit)
    return llvm::Type::Int64Ty;
    else
    return llvm::Type::Int32Ty;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstInitializer(Type* type, Initializer* init)
{
    LLConstant* _init = 0; // may return zero
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
        const LLType* ty = DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported const initializer: %s", init->toChars());
    }
    return _init;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstFieldInitializer(Type* t, Initializer* init)
{
    Logger::println("DtoConstFieldInitializer");
    LOG_SCOPE;

    const LLType* _type = DtoType(t);

    LLConstant* _init = DtoConstInitializer(t, init);
    assert(_init);
    if (_type != _init->getType())
    {
        Logger::cout() << "field init is: " << *_init << " type should be " << *_type << '\n';
        if (t->ty == Tsarray)
        {
            const llvm::ArrayType* arrty = isaArray(_type);
            uint64_t n = arrty->getNumElements();
            std::vector<LLConstant*> vals(n,_init);
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
            assert(ts->sym->ir.irStruct->constInit);
            _init = ts->sym->ir.irStruct->constInit;
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

LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var, llvm::BasicBlock* bb)
{
    LLSmallVector<LLValue*,2> v(2);
    v[0] = i0;
    v[1] = i1;
    return llvm::GetElementPtrInst::Create(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, const DStructIndexVector& src, const char* var, llvm::BasicBlock* bb)
{
    size_t n = src.size();
    LLSmallVector<LLValue*, 3> dst(n);

    size_t j=0;
    for (DStructIndexVector::const_iterator i=src.begin(); i!=src.end(); ++i)
        dst[j++] = DtoConstUint(*i);

    return llvm::GetElementPtrInst::Create(ptr, dst.begin(), dst.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, unsigned i, const char* var, llvm::BasicBlock* bb)
{
    return llvm::GetElementPtrInst::Create(ptr, llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var, llvm::BasicBlock* bb)
{
    LLSmallVector<LLValue*,2> v(2);
    v[0] = DtoConstUint(i0);
    v[1] = DtoConstUint(i1);
    return llvm::GetElementPtrInst::Create(ptr, v.begin(), v.end(), var, bb?bb:gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoNew(Type* newtype)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocmemoryT");
    // get type info
    LLConstant* ti = DtoTypeInfoOf(newtype);
    assert(isaPointer(ti));
    // call runtime allocator
    LLValue* mem = gIR->ir->CreateCall(fn, ti, ".gc_mem");
    // cast
    return DtoBitCast(mem, getPtrToType(DtoType(newtype)), ".gc_mem");
}

void DtoDeleteMemory(LLValue* ptr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delmemory");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(ptr, getVoidPtrType(), ".tmp"));
    // call
    llvm::CallInst::Create(fn, arg.begin(), arg.end(), "", gIR->scopebb());
}

void DtoDeleteClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delclass");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    llvm::CallInst::Create(fn, arg.begin(), arg.end(), "", gIR->scopebb());
}

void DtoDeleteInterface(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delinterface");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    llvm::CallInst::Create(fn, arg.begin(), arg.end(), "", gIR->scopebb());
}

void DtoDeleteArray(DValue* arr)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_delarray");
    // build args
    LLSmallVector<LLValue*,2> arg;
    arg.push_back(DtoArrayLen(arr));
    arg.push_back(DtoBitCast(DtoArrayPtr(arr), getVoidPtrType(), ".tmp"));
    // call
    llvm::CallInst::Create(fn, arg.begin(), arg.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoAssert(Loc* loc, DValue* msg)
{
    std::vector<LLValue*> args;
    LLConstant* c;

    // func
    const char* fname = msg ? "_d_assert_msg" : "_d_assert";
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, fname);

    // param attrs
    llvm::PAListPtr palist;
    int idx = 1;

    c = DtoConstString(loc->filename);

    // msg param
    if (msg)
    {
        if (DSliceValue* s = msg->isSlice())
        {
            llvm::AllocaInst* alloc = gIR->func()->msgArg;
            if (!alloc)
            {
                alloc = new llvm::AllocaInst(c->getType(), ".assertmsg", gIR->topallocapoint());
                DtoSetArray(alloc, DtoArrayLen(s), DtoArrayPtr(s));
                gIR->func()->msgArg = alloc;
            }
            args.push_back(alloc);
        }
        else
        {
            args.push_back(msg->getRVal());
        }
        palist = palist.addAttr(idx++, llvm::ParamAttr::ByVal);
    }

    // file param
    llvm::AllocaInst* alloc = gIR->func()->srcfileArg;
    if (!alloc)
    {
        alloc = new llvm::AllocaInst(c->getType(), ".srcfile", gIR->topallocapoint());
        gIR->func()->srcfileArg = alloc;
    }
    LLValue* ptr = DtoGEPi(alloc, 0,0, "tmp");
    DtoStore(c->getOperand(0), ptr);
    ptr = DtoGEPi(alloc, 0,1, "tmp");
    DtoStore(c->getOperand(1), ptr);

    args.push_back(alloc);
    palist = palist.addAttr(idx++, llvm::ParamAttr::ByVal);


    // line param
    c = DtoConstUint(loc->linnum);
    args.push_back(c);

    // call
    llvm::CallInst* call = llvm::CallInst::Create(fn, args.begin(), args.end(), "", gIR->scopebb());
    call->setParamAttrs(palist);
}

//////////////////////////////////////////////////////////////////////////////////////////

static const LLType* get_next_frame_ptr_type(Dsymbol* sc)
{
    assert(sc->isFuncDeclaration() || sc->isClassDeclaration());
    Dsymbol* p = sc->toParent2();
    if (!p->isFuncDeclaration() && !p->isClassDeclaration())
        Logger::println("unexpected parent symbol found while resolving frame pointer - '%s' kind: '%s'", p->toChars(), p->kind());
    assert(p->isFuncDeclaration() || p->isClassDeclaration());
    if (FuncDeclaration* fd = p->isFuncDeclaration())
    {
        LLValue* v = fd->ir.irFunc->nestedVar;
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

static LLValue* get_frame_ptr_impl(FuncDeclaration* func, Dsymbol* sc, LLValue* v)
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
            if (!func->ir.irFunc->nestedVar)
                return NULL;
            return DtoBitCast(v, func->ir.irFunc->nestedVar->getType());
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
            //idx += cd->ir.irStruct->interfaceVec.size();
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

static LLValue* get_frame_ptr(FuncDeclaration* func)
{
    Logger::println("Resolving context pointer for nested function: '%s'", func->toPrettyChars());
    LOG_SCOPE;
    IrFunction* irfunc = gIR->func();

    // in the right scope already
    if (func == irfunc->decl)
        return irfunc->decl->ir.irFunc->nestedVar;

    // use the 'this' pointer
    LLValue* ptr = irfunc->decl->ir.irFunc->thisVar;
    assert(ptr);

    // return the fully resolved frame pointer
    ptr = get_frame_ptr_impl(func, irfunc->decl, ptr);
    if (ptr) Logger::cout() << "Found context!" << *ptr;
    else Logger::cout() << "NULL context!\n";

    return ptr;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoNestedContext(FuncDeclaration* func)
{
    // resolve frame ptr
    LLValue* ptr = get_frame_ptr(func);
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

LLValue* DtoNestedVariable(VarDeclaration* vd)
{
    // log the frame list
    IrFunction* irfunc = gIR->func();
    if (Logger::enabled())
        print_nested_frame_list(vd, irfunc->decl);

    // resolve frame ptr
    FuncDeclaration* func = vd->toParent2()->isFuncDeclaration();
    assert(func);
    LLValue* ptr = DtoNestedContext(func);
    assert(ptr && "nested var, but no context");

    // we must cast here to be sure. nested classes just have a void*
    ptr = DtoBitCast(ptr, func->ir.irFunc->nestedVar->getType());

    // index nested var and load (if necessary)
    LLValue* v = DtoGEPi(ptr, 0, vd->ir.irLocal->nestedIndex, "tmp");
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
            assert(s->getType()->toBasetype() == lhs->getType()->toBasetype());
            DtoSetArray(lhs->getLVal(),DtoArrayLen(s),DtoArrayPtr(s));
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
        if (DtoType(lhs->getType()) == DtoType(rhs->getType())) {
            DtoStaticArrayCopy(lhs->getLVal(), rhs->getRVal());
        }
        else {
            DtoArrayInit(lhs->getLVal(), rhs->getRVal());
        }
    }
    else if (t->ty == Tdelegate) {
        if (rhs->isNull())
            DtoDelegateToNull(lhs->getLVal());
        else if (!rhs->inPlace()) {
            LLValue* l = lhs->getLVal();
            LLValue* r = rhs->getRVal();
            Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
            DtoDelegateCopy(l, r);
        }
    }
    else if (t->ty == Tclass) {
        assert(t2->ty == Tclass);
        // assignment to this in constructor special case
        if (lhs->isThis()) {
            LLValue* tmp = rhs->getRVal();
            FuncDeclaration* fdecl = gIR->func()->decl;
            // respecify the this param
            if (!llvm::isa<llvm::AllocaInst>(fdecl->ir.irFunc->thisVar))
                fdecl->ir.irFunc->thisVar = new llvm::AllocaInst(tmp->getType(), "newthis", gIR->topallocapoint());
            DtoStore(tmp, fdecl->ir.irFunc->thisVar);
        }
        // regular class ref -> class ref assignment
        else {
            DtoStore(rhs->getRVal(), lhs->getLVal());
        }
    }
    else if (t->iscomplex()) {
        assert(!lhs->isComplex());

        LLValue* dst;
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
        LLValue* l = lhs->getLVal();
        LLValue* r = rhs->getRVal();
        Logger::cout() << "assign\nlhs: " << *l << "rhs: " << *r << '\n';
        const LLType* lit = l->getType()->getContainedType(0);
        if (r->getType() != lit) {
            // handle lvalue cast assignments
            if (DLRValue* lr = lhs->isLRValue()) {
                Logger::println("lvalue cast!");
                r = DtoCast(rhs, lr->getLType())->getRVal();
            }
            else {
                r = DtoCast(rhs, lhs->getType())->getRVal();
            }
            Logger::cout() << "really assign\nlhs: " << *l << "rhs: " << *r << '\n';
            assert(r->getType() == l->getType()->getContainedType(0));
        }
        gIR->ir->CreateStore(r, l);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
DValue* DtoCastInt(DValue* val, Type* _to)
{
    const LLType* tolltype = DtoType(_to);

    Type* to = DtoDType(_to);
    Type* from = DtoDType(val->getType());
    assert(from->isintegral());

    size_t fromsz = from->size();
    size_t tosz = to->size();

    LLValue* rval = val->getRVal();
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
            rval = DtoBitCast(rval, tolltype);
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
    const LLType* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(val->getType());
    assert(fromtype->ty == Tpointer);

    LLValue* rval;

    if (totype->ty == Tpointer || totype->ty == Tclass) {
        LLValue* src = val->getRVal();
        Logger::cout() << "src: " << *src << "to type: " << *tolltype << '\n';
        rval = DtoBitCast(src, tolltype);
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

    const LLType* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(val->getType());
    assert(fromtype->isfloating());

    size_t fromsz = fromtype->size();
    size_t tosz = totype->size();

    LLValue* rval;

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
        const LLType* toty = DtoComplexBaseType(to);

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
        LLValue* mem = new llvm::AllocaInst(DtoType(_to), "castcomplextmp", gIR->topallocapoint());
        DtoComplexSet(mem, re, im);
        return new DLRValue(val->getType(), val->getRVal(), _to, mem);
    }
    else if (to->isimaginary()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->im);
        LLValue* v = val->getRVal();
        DImValue* im = new DImValue(to, DtoLoad(DtoGEPi(v,0,1,"tmp")));
        return DtoCastFloat(im, to);
    }
    else if (to->isfloating()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->re);
        LLValue* v = val->getRVal();
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
LLConstant* DtoConstBool(bool b)
{
    return llvm::ConstantInt::get(llvm::Type::Int1Ty, b, false);
}

llvm::ConstantFP* DtoConstFP(Type* t, long double value)
{
    TY ty = DtoDType(t)->ty;
    if (ty == Tfloat32 || ty == Timaginary32)
        return llvm::ConstantFP::get(llvm::APFloat(float(value)));
    else if (ty == Tfloat64 || ty == Timaginary64 || ty == Tfloat80 || ty == Timaginary80)
        return llvm::ConstantFP::get(llvm::APFloat(double(value)));
}


//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstString(const char* str)
{
    std::string s(str);
    LLConstant* init = llvm::ConstantArray::get(s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, "stringliteral", gIR->module);
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
        init->getType(), true,llvm::GlobalValue::InternalLinkage, init, "stringliteral", gIR->module);
    if (section) gvar->setSection(section);
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    return llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(LLValue* dst, LLValue* nbytes)
{
    const LLType* arrty = getPtrToType(llvm::Type::Int8Ty);
    llvm::Value *dstarr;
    if (dst->getType() == arrty)
    {
        dstarr = dst;
    }
    else
    {
        dstarr = DtoBitCast(dst,arrty);
    }

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemSet64() : LLVM_DeclareMemSet32();
    std::vector<LLValue*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = nbytes;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::CallInst::Create(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes)
{
    const LLType* arrty = getVoidPtrType();

    LLValue* dstarr;
    if (dst->getType() == arrty)
        dstarr = dst;
    else
        dstarr = DtoBitCast(dst, arrty, "tmp");

    LLValue* srcarr;
    if (src->getType() == arrty)
        srcarr = src;
    else
        srcarr = DtoBitCast(src, arrty, "tmp");

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemCpy64() : LLVM_DeclareMemCpy32();
    std::vector<LLValue*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = nbytes;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::CallInst::Create(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
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
        return ptr->getType()->getContainedType(0)->isFirstClassType();
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

const llvm::PointerType* isaPointer(LLValue* v)
{
    return llvm::dyn_cast<llvm::PointerType>(v->getType());
}

const llvm::PointerType* isaPointer(const LLType* t)
{
    return llvm::dyn_cast<llvm::PointerType>(t);
}

const llvm::ArrayType* isaArray(LLValue* v)
{
    return llvm::dyn_cast<llvm::ArrayType>(v->getType());
}

const llvm::ArrayType* isaArray(const LLType* t)
{
    return llvm::dyn_cast<llvm::ArrayType>(t);
}

const llvm::StructType* isaStruct(LLValue* v)
{
    return llvm::dyn_cast<llvm::StructType>(v->getType());
}

const llvm::StructType* isaStruct(const LLType* t)
{
    return llvm::dyn_cast<llvm::StructType>(t);
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

const llvm::PointerType* getPtrToType(const LLType* t)
{
    return llvm::PointerType::get(t, 0);
}

const llvm::PointerType* getVoidPtrType()
{
    return getPtrToType(llvm::Type::Int8Ty);
}

llvm::ConstantPointerNull* getNullPtr(const LLType* t)
{
    const llvm::PointerType* pt = llvm::cast<llvm::PointerType>(t);
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

void DtoLazyStaticInit(bool istempl, LLValue* gvar, Initializer* init, Type* t)
{
    // create a flag to make sure initialization only happens once
    llvm::GlobalValue::LinkageTypes gflaglink = istempl ? llvm::GlobalValue::WeakLinkage : llvm::GlobalValue::InternalLinkage;
    std::string gflagname(gvar->getName());
    gflagname.append("__initflag");
    llvm::GlobalVariable* gflag = new llvm::GlobalVariable(llvm::Type::Int1Ty,false,gflaglink,DtoConstBool(false),gflagname,gIR->module);

    // check flag and do init if not already done
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* initbb = llvm::BasicBlock::Create("ifnotinit",gIR->topfunc(),oldend);
    llvm::BasicBlock* endinitbb = llvm::BasicBlock::Create("ifnotinitend",gIR->topfunc(),oldend);
    LLValue* cond = gIR->ir->CreateICmpEQ(gIR->ir->CreateLoad(gflag,"tmp"),DtoConstBool(false));
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
    if (vd->ir.initialized) return;
    vd->ir.initialized = gIR->dmodule;

    Logger::println("* DtoConstInitGlobal(%s)", vd->toChars());
    LOG_SCOPE;

    bool emitRTstaticInit = false;

    LLConstant* _init = 0;
    if (vd->parent && vd->parent->isFuncDeclaration() && vd->init && vd->init->isExpInitializer()) {
        _init = DtoConstInitializer(vd->type, NULL);
        emitRTstaticInit = true;
    }
    else {
        _init = DtoConstInitializer(vd->type, vd->init);
    }

    const LLType* _type = DtoType(vd->type);
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
            assert(ts->sym->ir.irStruct->constInit);
            _init = ts->sym->ir.irStruct->constInit;
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
    llvm::cast<llvm::OpaqueType>(vd->ir.irGlobal->type.get())->refineAbstractTypeTo(_type);
    _type = vd->ir.irGlobal->type.get();
    //_type->dump();
    assert(!_type->isAbstract());

    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(vd->ir.irGlobal->value);
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
    if (dsym->ir.declared) return;
    Logger::println("DtoForceDeclareDsymbol(%s)", dsym->toPrettyChars());
    LOG_SCOPE;
    DtoResolveDsymbol(dsym);

    DtoEmptyResolveList();

    DtoDeclareDsymbol(dsym);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoForceConstInitDsymbol(Dsymbol* dsym)
{
    if (dsym->ir.initialized) return;
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
    if (dsym->ir.defined) return;
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

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoInterfaceInfoType()
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
    const LLType* byteptrptrty = getPtrToType(getPtrToType(llvm::Type::Int8Ty));
    vtbltypes.push_back(byteptrptrty);
    types.push_back(llvm::StructType::get(vtbltypes));
    // int offset
    types.push_back(llvm::Type::Int32Ty);
    // create type
    gIR->interfaceInfoType = llvm::StructType::get(types);

    return gIR->interfaceInfoType;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoTypeInfoOf(Type* type, bool base)
{
    const LLType* typeinfotype = DtoType(Type::typeinfo->type);
    if (!type->vtinfo)
        type->getTypeInfo(NULL);
    TypeInfoDeclaration* tidecl = type->vtinfo;
    DtoForceDeclareDsymbol(tidecl);
    assert(tidecl->ir.irGlobal != NULL);
    LLConstant* c = isaConstant(tidecl->ir.irGlobal->value);
    assert(c != NULL);
    if (base)
        return llvm::ConstantExpr::getBitCast(c, typeinfotype);
    return c;
}












