//===-- tollvm.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/tollvm.h"
#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "id.h"
#include "init.h"
#include "module.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/typeinf.h"
#include "ir/irtype.h"
#include "ir/irtypeclass.h"
#include "ir/irtypefunction.h"
#include "ir/irtypestruct.h"

bool DtoIsPassedByRef(Type* type)
{
    Type* typ = type->toBasetype();
    TY t = typ->ty;
    return (t == Tstruct || t == Tsarray);
}

RET retStyle(TypeFunction *tf)
{
    bool sret = gABI->returnInArg(tf);
    return sret ? RETstack : RETregs;
}

bool DtoIsReturnInArg(CallExp *ce)
{
    TypeFunction *tf = static_cast<TypeFunction *>(ce->e1->type->toBasetype());
    if (tf->ty == Tfunction && (!ce->f || ce->f->llvmInternal != LLVMintrinsic))
        return retStyle(tf) == RETstack;
    return false;
}

AttrBuilder::A DtoShouldExtend(Type* type)
{
    type = type->toBasetype();
    if (type->isintegral())
    {
        switch(type->ty)
        {
        case Tint8:
        case Tint16:
            return LDC_ATTRIBUTE(SExt);

        case Tuns8:
        case Tuns16:
            return LDC_ATTRIBUTE(ZExt);

        default:
            // Do not extend.
            break;
        }
    }

    return LDC_ATTRIBUTE(None);
}

LLType* DtoType(Type* t)
{
    t = stripModifiers( t );

    if (t->ctype)
    {
        return t->ctype->getLLType();
    }

    IF_LOG Logger::println("Building type: %s", t->toChars());
    LOG_SCOPE;

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
    case Tint128:
    case Tuns128:
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
        return IrTypeBasic::get(t)->getLLType();
    }

    // pointers
    case Tnull:
    case Tpointer:
    {
        return IrTypePointer::get(t)->getLLType();
    }

    // arrays
    case Tarray:
    {
        return IrTypeArray::get(t)->getLLType();
    }

    case Tsarray:
    {
        return IrTypeSArray::get(t)->getLLType();
    }

    // aggregates
    case Tstruct:
    {
        TypeStruct* ts = static_cast<TypeStruct*>(t);
        if (ts->sym->type->ctype)
        {
            // This should not happen, but the frontend seems to be buggy. Not
            // sure if this is the best way to handle the situation, but we
            // certainly don't want to override ts->sym->type->ctype.
            IF_LOG Logger::cout() << "Struct with multiple Types detected: " <<
                ts->toChars() << " (" << ts->sym->locToChars() << ")" << std::endl;
            return ts->sym->type->ctype->getLLType();
        }
        return IrTypeStruct::get(ts->sym)->getLLType();
    }
    case Tclass:
    {
        TypeClass* tc = static_cast<TypeClass*>(t);
        if (tc->sym->type->ctype)
        {
            // See Tstruct case.
            IF_LOG Logger::cout() << "Class with multiple Types detected: " <<
                tc->toChars() << " (" << tc->sym->locToChars() << ")" << std::endl;
            return tc->sym->type->ctype->getLLType();
        }
        return IrTypeClass::get(tc->sym)->getLLType();
    }

    // functions
    case Tfunction:
    {
        return IrTypeFunction::get(t)->getLLType();
    }

    // delegates
    case Tdelegate:
    {
        return IrTypeDelegate::get(t)->getLLType();
    }

    // typedefs
    // enum

    // FIXME: maybe just call toBasetype first ?
    case Tenum:
    {
        Type* bt = t->toBasetype();
        assert(bt);
        return DtoType(bt);
    }

    // associative arrays
    case Taarray:
        return getVoidPtrType();

    case Tvector:
    {
        return IrTypeVector::get(t)->getLLType();
    }

/*
    Not needed atm as VarDecls for tuples are rewritten as a string of
    VarDecls for the fields (u -> _u_field_0, ...)

    case Ttuple:
    {
        TypeTuple* ttupl = static_cast<TypeTuple*>(t);
        return DtoStructTypeFromArguments(ttupl->arguments);
    }
*/

    default:
        llvm_unreachable("Unknown class of D Type!");
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
LLType* DtoStructTypeFromArguments(Arguments* arguments)
{
    if (!arguments)
        return LLType::getVoidTy(gIR->context());

    std::vector<LLType*> types;
    for (size_t i = 0; i < arguments->dim; i++)
    {
        Argument *arg = (*arguments)[i];
        assert(arg && arg->type);

        types.push_back(DtoType(arg->type));
    }
    return LLStructType::get(types);
}
*/

//////////////////////////////////////////////////////////////////////////////////////////

LLType* voidToI8(LLType* t)
{
    if (t == LLType::getVoidTy(gIR->context()))
        return LLType::getInt8Ty(gIR->context());
    return t;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLType* i1ToI8(LLType* t)
{
    if (t == LLType::getInt1Ty(gIR->context()))
        return LLType::getInt8Ty(gIR->context());
    return t;
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
    b1 = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,l,r);

    l = gIR->ir->CreateExtractValue(lhs, 1);
    r = gIR->ir->CreateExtractValue(rhs, 1);
    b2 = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ,l,r);

    LLValue* b = gIR->ir->CreateAnd(b1,b2);

    if (op == TOKnotequal || op == TOKnotidentity)
        return gIR->ir->CreateNot(b);

    return b;
}

//////////////////////////////////////////////////////////////////////////////////////////

LinkageWithCOMDAT DtoLinkage(Dsymbol* sym)
{
    if (DtoIsTemplateInstance(sym))
        return LinkageWithCOMDAT(templateLinkage, supportsCOMDAT());
    return LinkageWithCOMDAT(llvm::GlobalValue::ExternalLinkage, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLIntegerType* DtoSize_t()
{
    // the type of size_t does not change once set
    static LLIntegerType* t = NULL;
    if (t == NULL)
        t = (global.params.isLP64) ? LLType::getInt64Ty(gIR->context()) : LLType::getInt32Ty(gIR->context());
    return t;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP1(LLValue* ptr, LLValue* i0, const char* var, llvm::BasicBlock* bb)
{
    LLPointerType* p = isaPointer(ptr);
    assert(p && "GEP expects a pointer type");
    return llvm::GetElementPtrInst::Create(
#if LDC_LLVM_VER >= 307
        p->getElementType(),
#endif
        ptr, i0, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var, llvm::BasicBlock* bb)
{
    LLPointerType* p = isaPointer(ptr);
    assert(p && "GEP expects a pointer type");
    LLValue* v[] = { i0, i1 };
    return llvm::GetElementPtrInst::Create(
#if LDC_LLVM_VER >= 307
        p->getElementType(),
#endif
        ptr, v, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi1(LLValue* ptr, unsigned i, const char* var, llvm::BasicBlock* bb)
{
    LLPointerType* p = isaPointer(ptr);
    assert(p && "GEP expects a pointer type");
    return llvm::GetElementPtrInst::Create(
#if LDC_LLVM_VER >= 307
        p->getElementType(),
#endif
        ptr, DtoConstUint(i), var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var, llvm::BasicBlock* bb)
{
    LLPointerType* p = isaPointer(ptr);
    assert(p && "GEP expects a pointer type");
    LLValue* v[] = { DtoConstUint(i0), DtoConstUint(i1) };
    return llvm::GetElementPtrInst::Create(
#if LDC_LLVM_VER >= 307
        p->getElementType(),
#endif
        ptr, v, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoGEPi(LLConstant* ptr, unsigned i0, unsigned i1)
{
    LLPointerType* p = isaPointer(ptr);
    assert(p && "GEP expects a pointer type");
    LLValue* v[] = { DtoConstUint(i0), DtoConstUint(i1) };
    return llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
        p->getElementType(),
#endif
        ptr, v, true);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemSet(LLValue* dst, LLValue* val, LLValue* nbytes)
{
    LLType* VoidPtrTy = getVoidPtrType();

    dst = DtoBitCast(dst, VoidPtrTy);

    gIR->ir->CreateMemSet(dst, val, nbytes, 1 /*Align*/, false /*isVolatile*/);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemSetZero(LLValue* dst, LLValue* nbytes)
{
    DtoMemSet(dst, DtoConstUbyte(0), nbytes);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes, unsigned align)
{
    LLType* VoidPtrTy = getVoidPtrType();

    dst = DtoBitCast(dst, VoidPtrTy);
    src = DtoBitCast(src, VoidPtrTy);

    gIR->ir->CreateMemCpy(dst, src, nbytes, align, false /*isVolatile*/);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoMemCmp(LLValue* lhs, LLValue* rhs, LLValue* nbytes)
{
    // int memcmp ( const void * ptr1, const void * ptr2, size_t num );

    LLType* VoidPtrTy = getVoidPtrType();
    LLFunction* fn = gIR->module.getFunction("memcmp");
    if (!fn)
    {
        LLType* Tys[] = { VoidPtrTy, VoidPtrTy, DtoSize_t() };
        LLFunctionType* fty = LLFunctionType::get(LLType::getInt32Ty(gIR->context()),
                                                  Tys, false);
        fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, "memcmp", &gIR->module);
    }

    lhs = DtoBitCast(lhs, VoidPtrTy);
    rhs = DtoBitCast(rhs, VoidPtrTy);

#if LDC_LLVM_VER >= 307
    return gIR->ir->CreateCall(fn, { lhs, rhs, nbytes });
#else
    return gIR->ir->CreateCall3(fn, lhs, rhs, nbytes);
#endif
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

llvm::ConstantInt* DtoConstSize_t(uint64_t i)
{
    return LLConstantInt::get(DtoSize_t(), i, false);
}
llvm::ConstantInt* DtoConstUint(unsigned i)
{
    return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, false);
}
llvm::ConstantInt* DtoConstInt(int i)
{
    return LLConstantInt::get(LLType::getInt32Ty(gIR->context()), i, true);
}
LLConstant* DtoConstBool(bool b)
{
    return LLConstantInt::get(LLType::getInt1Ty(gIR->context()), b, false);
}
llvm::ConstantInt* DtoConstUbyte(unsigned char i)
{
    return LLConstantInt::get(LLType::getInt8Ty(gIR->context()), i, false);
}

LLConstant* DtoConstFP(Type* t, longdouble value)
{
    LLType* llty = DtoType(t);
    assert(llty->isFloatingPointTy());

    if(llty == LLType::getFloatTy(gIR->context()) || llty == LLType::getDoubleTy(gIR->context()))
        return LLConstantFP::get(llty, value);
    else if(llty == LLType::getX86_FP80Ty(gIR->context())) {
        uint64_t bits[] = { 0, 0 };
        bits[0] = *reinterpret_cast<uint64_t*>(&value);
        bits[1] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint64_t*>(&value) + 1);
#if LDC_LLVM_VER >= 303
        return LLConstantFP::get(gIR->context(), APFloat(APFloat::x87DoubleExtended, APInt(80, 2, bits)));
#else
        return LLConstantFP::get(gIR->context(), APFloat(APInt(80, 2, bits)));
#endif
    } else if(llty == LLType::getPPC_FP128Ty(gIR->context())) {
        uint64_t bits[] = {0, 0};
        bits[0] = *reinterpret_cast<uint64_t*>(&value);
        bits[1] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint64_t*>(&value) + 1);
#if LDC_LLVM_VER >= 303
        return LLConstantFP::get(gIR->context(), APFloat(APFloat::PPCDoubleDouble, APInt(128, 2, bits)));
#else
        return LLConstantFP::get(gIR->context(), APFloat(APInt(128, 2, bits)));
#endif
    }

    llvm_unreachable("Unknown floating point type encountered");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstString(const char* str)
{
    llvm::StringRef s(str ? str : "");
    llvm::GlobalVariable* gvar = (gIR->stringLiteral1ByteCache.find(s) ==
                                  gIR->stringLiteral1ByteCache.end())
                                 ? 0 : gIR->stringLiteral1ByteCache[s];
    if (gvar == 0)
    {
        llvm::Constant* init = llvm::ConstantDataArray::getString(gIR->context(), s, true);
        gvar = new llvm::GlobalVariable(gIR->module, init->getType(), true,
                                        llvm::GlobalValue::PrivateLinkage, init, ".str");
        gvar->setUnnamedAddr(true);
        gIR->stringLiteral1ByteCache[s] = gvar;
    }
    LLConstant* idxs[] = { DtoConstUint(0), DtoConstUint(0) };
    return DtoConstSlice(
        DtoConstSize_t(s.size()),
        llvm::ConstantExpr::getGetElementPtr(
#if LDC_LLVM_VER >= 307
            gvar->getInitializer()->getType(),
#endif
            gvar, idxs, true),
        Type::tchar->arrayOf()
    );
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoLoad(LLValue* src, const char* name)
{
//     if (Logger::enabled())
//         Logger::cout() << "loading " << *src <<  '\n';
    llvm::LoadInst* ld = gIR->ir->CreateLoad(src, name);
    //ld->setVolatile(gIR->func()->inVolatile);
    return ld;
}

// Like DtoLoad, but the pointer is guaranteed to be aligned appropriately for the type.
LLValue* DtoAlignedLoad(LLValue* src, const char* name)
{
    llvm::LoadInst* ld = gIR->ir->CreateLoad(src, name);
    ld->setAlignment(getABITypeAlign(ld->getType()));
    return ld;
}

LLValue* DtoVolatileLoad(LLValue* src, const char* name)
{
    llvm::LoadInst* ld = gIR->ir->CreateLoad(src, name);
    ld->setVolatile(true);
    return ld;
}


void DtoStore(LLValue* src, LLValue* dst)
{
    assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
        "Should store bools as i8 instead of i1.");
    gIR->ir->CreateStore(src,dst);
}

void DtoVolatileStore(LLValue* src, LLValue* dst)
{
    assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
        "Should store bools as i8 instead of i1.");
    gIR->ir->CreateStore(src, dst)->setVolatile(true);
}

void DtoStoreZextI8(LLValue* src, LLValue* dst)
{
    if (src->getType() == llvm::Type::getInt1Ty(gIR->context()))
    {
        llvm::Type* i8 = llvm::Type::getInt8Ty(gIR->context());
        assert(dst->getType()->getContainedType(0) == i8);
        src = gIR->ir->CreateZExt(src, i8);
    }
    gIR->ir->CreateStore(src, dst);
}

// Like DtoStore, but the pointer is guaranteed to be aligned appropriately for the type.
void DtoAlignedStore(LLValue* src, LLValue* dst)
{
    assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
        "Should store bools as i8 instead of i1.");
    llvm::StoreInst* st = gIR->ir->CreateStore(src,dst);
    st->setAlignment(getABITypeAlign(src->getType()));
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoBitCast(LLValue* v, LLType* t, const char* name)
{
    if (v->getType() == t)
        return v;
    assert(!isaStruct(t));
    return gIR->ir->CreateBitCast(v, t, name);
}

LLConstant* DtoBitCast(LLConstant* v, LLType* t)
{
    if (v->getType() == t)
        return v;
    return llvm::ConstantExpr::getBitCast(v, t);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoInsertValue(LLValue* aggr, LLValue* v, unsigned idx, const char* name)
{
    return gIR->ir->CreateInsertValue(aggr, v, idx, name);
}

LLValue* DtoExtractValue(LLValue* aggr, unsigned idx, const char* name)
{
    return gIR->ir->CreateExtractValue(aggr, idx, name);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoInsertElement(LLValue* vec, LLValue* v, LLValue *idx, const char* name)
{
    return gIR->ir->CreateInsertElement(vec, v, idx, name);
}

LLValue* DtoExtractElement(LLValue* vec, LLValue *idx, const char* name)
{
    return gIR->ir->CreateExtractElement(vec, idx, name);
}

LLValue* DtoInsertElement(LLValue* vec, LLValue* v, unsigned idx, const char* name)
{
    return DtoInsertElement(vec, v, DtoConstUint(idx), name);
}

LLValue* DtoExtractElement(LLValue* vec, unsigned idx, const char* name)
{
    return DtoExtractElement(vec, DtoConstUint(idx), name);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLPointerType* isaPointer(LLValue* v)
{
    return llvm::dyn_cast<LLPointerType>(v->getType());
}

LLPointerType* isaPointer(LLType* t)
{
    return llvm::dyn_cast<LLPointerType>(t);
}

LLArrayType* isaArray(LLValue* v)
{
    return llvm::dyn_cast<LLArrayType>(v->getType());
}

LLArrayType* isaArray(LLType* t)
{
    return llvm::dyn_cast<LLArrayType>(t);
}

LLStructType* isaStruct(LLValue* v)
{
    return llvm::dyn_cast<LLStructType>(v->getType());
}

LLStructType* isaStruct(LLType* t)
{
    return llvm::dyn_cast<LLStructType>(t);
}

LLFunctionType* isaFunction(LLValue* v)
{
    return llvm::dyn_cast<LLFunctionType>(v->getType());
}

LLFunctionType* isaFunction(LLType* t)
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

LLPointerType* getPtrToType(LLType* t)
{
    if (t == LLType::getVoidTy(gIR->context()))
        t = LLType::getInt8Ty(gIR->context());
    return LLPointerType::get(t, 0);
}

LLPointerType* getVoidPtrType()
{
    return getPtrToType(LLType::getInt8Ty(gIR->context()));
}

llvm::ConstantPointerNull* getNullPtr(LLType* t)
{
    LLPointerType* pt = llvm::cast<LLPointerType>(t);
    return llvm::ConstantPointerNull::get(pt);
}

LLConstant* getNullValue(LLType* t)
{
    return LLConstant::getNullValue(t);
}

//////////////////////////////////////////////////////////////////////////////////////////

size_t getTypeBitSize(LLType* t)
{
    return gDataLayout->getTypeSizeInBits(t);
}

size_t getTypeStoreSize(LLType* t)
{
    return gDataLayout->getTypeStoreSize(t);
}

size_t getTypePaddedSize(LLType* t)
{
    size_t sz = gDataLayout->getTypeAllocSize(t);
    //Logger::cout() << "abi type size of: " << *t << " == " << sz << '\n';
    return sz;
}

size_t getTypeAllocSize(LLType* t)
{
    return gDataLayout->getTypeAllocSize(t);
}

unsigned char getABITypeAlign(LLType* t)
{
    return gDataLayout->getABITypeAlignment(t);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLStructType* DtoMutexType()
{
    if (gIR->mutexType)
        return gIR->mutexType;

    // The structures defined here must be the same as in druntime/src/rt/critical.c

    // Windows
    if (global.params.targetTriple.isOSWindows())
    {
        llvm::Type *VoidPtrTy = llvm::Type::getInt8PtrTy(gIR->context());
        llvm::Type *Int32Ty = llvm::Type::getInt32Ty(gIR->context());

        // Build RTL_CRITICAL_SECTION; size is 24 (32bit) or 40 (64bit)
        LLType *rtl_types[] = {
            VoidPtrTy, // Pointer to DebugInfo
            Int32Ty,   // LockCount
            Int32Ty,   // RecursionCount
            VoidPtrTy, // Handle of OwningThread
            VoidPtrTy, // Handle of LockSemaphore
            VoidPtrTy  // SpinCount
        };
        LLStructType* rtl = LLStructType::create(gIR->context(), rtl_types, "RTL_CRITICAL_SECTION");

        // Build D_CRITICAL_SECTION; size is 28 (32bit) or 48 (64bit)
        LLStructType *mutex = LLStructType::create(gIR->context(), "D_CRITICAL_SECTION");
        LLType *types[] = { getPtrToType(mutex), rtl };
        mutex->setBody(types);

        // Cache type
        gIR->mutexType = mutex;

        return mutex;
    }

    // FreeBSD
    else if (global.params.targetTriple.getOS() == llvm::Triple::FreeBSD) {
        // Just a pointer
        return LLStructType::get(gIR->context(), DtoSize_t());
    }

    // pthread_fastlock
    LLType *types2[] = {
        DtoSize_t(),
        LLType::getInt32Ty(gIR->context()) 
    };
    LLStructType* fastlock = LLStructType::get(gIR->context(), types2, false);

    // pthread_mutex
    LLType *types1[] = {
        LLType::getInt32Ty(gIR->context()),
        LLType::getInt32Ty(gIR->context()),
        getVoidPtrType(),
        LLType::getInt32Ty(gIR->context()),
        fastlock
    };
    LLStructType* pmutex = LLStructType::get(gIR->context(), types1, false);

    // D_CRITICAL_SECTION
    LLStructType* mutex = LLStructType::create(gIR->context(), "D_CRITICAL_SECTION");
    LLType *types[] = { getPtrToType(mutex), pmutex };
    mutex->setBody(types);

    // Cache type
    gIR->mutexType = mutex;

    return pmutex;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLStructType* DtoModuleReferenceType()
{
    if (gIR->moduleRefType)
        return gIR->moduleRefType;

    // this is a recursive type so start out with a struct without body
    LLStructType* st = LLStructType::create(gIR->context(), "ModuleReference");

    // add members
    LLType *types[] = {
        getPtrToType(st),
        DtoType(Module::moduleinfo->type->pointerTo())
    };

    // resolve type
    st->setBody(types);

    // done
    gIR->moduleRefType = st;
    return st;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoAggrPair(LLType* type, LLValue* V1, LLValue* V2, const char* name)
{
    LLValue* res = llvm::UndefValue::get(type);
    res = gIR->ir->CreateInsertValue(res, V1, 0);
    return gIR->ir->CreateInsertValue(res, V2, 1, name);
}

LLValue* DtoAggrPair(LLValue* V1, LLValue* V2, const char* name)
{
    LLType *types[] = {  V1->getType(), V2->getType() };
    LLType *t = LLStructType::get(gIR->context(), types, false);
    return DtoAggrPair(t, V1, V2, name);
}

LLValue* DtoAggrPaint(LLValue* aggr, LLType* as)
{
    if (aggr->getType() == as)
        return aggr;

    LLValue* res = llvm::UndefValue::get(as);

    LLValue* V = gIR->ir->CreateExtractValue(aggr, 0);
    V = DtoBitCast(V, as->getContainedType(0));
    res = gIR->ir->CreateInsertValue(res, V, 0);

    V = gIR->ir->CreateExtractValue(aggr, 1);
    V = DtoBitCast(V, as->getContainedType(1));
    return gIR->ir->CreateInsertValue(res, V, 1);
}
