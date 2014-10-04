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
    gABI->newFunctionType(tf);
    bool retInArg = gABI->returnInArg(tf);
    gABI->doneWithFunctionType();
    return retInArg ? RETstack : RETregs;
}

bool DtoIsReturnInArg(CallExp *ce)
{
    TypeFunction *tf = static_cast<TypeFunction *>(ce->e1->type->toBasetype());
    if (tf->ty == Tfunction && (!ce->f || ce->f->llvmInternal != LLVMintrinsic))
        return retStyle(tf) == RETstack;
    return false;
}

#if LDC_LLVM_VER >= 303
llvm::Attribute::AttrKind DtoShouldExtend(Type* type)
#elif LDC_LLVM_VER == 302
llvm::Attributes::AttrVal DtoShouldExtend(Type* type)
#else
llvm::Attributes DtoShouldExtend(Type* type)
#endif
{
    type = type->toBasetype();
    if (type->isintegral())
    {
        switch(type->ty)
        {
        case Tint8:
        case Tint16:
#if LDC_LLVM_VER >= 303
            return llvm::Attribute::SExt;
#elif LDC_LLVM_VER == 302
            return llvm::Attributes::SExt;
#else
            return llvm::Attribute::SExt;
#endif

        case Tuns8:
        case Tuns16:
#if LDC_LLVM_VER >= 303
            return llvm::Attribute::ZExt;
#elif LDC_LLVM_VER == 302
            return llvm::Attributes::ZExt;
#else
            return llvm::Attribute::ZExt;
#endif
        default:
            // Do not extend.
            break;
        }
    }
#if LDC_LLVM_VER >= 303
    return llvm::Attribute::None;
#elif LDC_LLVM_VER == 302
    return llvm::Attributes::None;
#else
    return llvm::Attribute::None;
#endif
}

LLType* DtoType(Type* t)
{
    t = stripModifiers( t );

    if (t->irtype)
    {
        return t->irtype->getLLType();
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
        if (ts->sym->type->irtype)
        {
            // This should not happen, but the frontend seems to be buggy. Not
            // sure if this is the best way to handle the situation, but we
            // certainly don't want to override ts->sym->type->irtype.
            IF_LOG Logger::cout() << "Struct with multiple Types detected: " <<
                ts->toChars() << " (" << ts->sym->locToChars() << ")" << std::endl;
            return ts->sym->type->irtype->getLLType();
        }
        return IrTypeStruct::get(ts->sym)->getLLType();
    }
    case Tclass:
    {
        TypeClass* tc = static_cast<TypeClass*>(t);
        if (tc->sym->type->irtype)
        {
            // See Tstruct case.
            IF_LOG Logger::cout() << "Class with multiple Types detected: " <<
                tc->toChars() << " (" << tc->sym->locToChars() << ")" << std::endl;
            return tc->sym->type->irtype->getLLType();
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
        Argument *arg = static_cast<Argument *>(arguments->data[i]);
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

LLGlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym)
{
    // global/static variable
    if (VarDeclaration* vd = sym->isVarDeclaration())
    {
        // template
        if (DtoIsTemplateInstance(sym))
            return templateLinkage;

        // Currently, we have to consider all variables, even function-local
        // statics, to be external, as CTFE might cause template functions
        // instances to be semantic3'd that occur within the body of a function
        // from an imported module. Consequently, a copy of them is codegen'd
        // in the importing module, even if they might reference a static in a
        // function in the imported module (e.g. via an alias parameter).
        //
        // A fix for this would be to track instantiations/semantic3 runs made
        // solely for CTFE purposes in a way similar to how the extra inlining
        // semantic runs are handled.
        //
        // LDC_FIXME: Can this also occur for functions? Find a better solution.
        if (true || vd->storage_class & STCextern)
            return llvm::GlobalValue::ExternalLinkage;

    }
    else if (FuncDeclaration* fdecl = sym->isFuncDeclaration())
    {
        assert(fdecl->type->ty == Tfunction);
        TypeFunction* ft = static_cast<TypeFunction*>(fdecl->type);

        // intrinsics are always external
        if (fdecl->llvmInternal == LLVMintrinsic)
            return llvm::GlobalValue::ExternalLinkage;

        // array operations are always template linkage
        if (fdecl->isArrayOp  && !isDruntimeArrayOp(fdecl))
            return templateLinkage;

        // template instances should have weak linkage
        // but only if there's a body, and it's not naked
        // otherwise we make it external
        if (DtoIsTemplateInstance(fdecl) && fdecl->fbody && !fdecl->naked)
            return templateLinkage;

        // extern(C) functions are always external
        if (ft->linkage == LINKc)
            return llvm::GlobalValue::ExternalLinkage;

        // If a function without a body is nested in another
        // function, we cannot use internal linkage for that
        // function (see below about nested functions)
        // FIXME: maybe there is a better way without emission
        // of needless symbols?
        if (!fdecl->fbody)
            return llvm::GlobalValue::ExternalLinkage;
    }
    // class
    else if (ClassDeclaration* cd = sym->isClassDeclaration())
    {
        // template
        if (DtoIsTemplateInstance(cd))
            return templateLinkage;
    }
    else
    {
        llvm_unreachable("not global/function");
    }

    // The logic here should be sound in theory, but as long as the frontend
    // keeps inserting templates into wrong modules, this yields to linking
    // errors (see e.g. GitHub issue #558).
#if 0
    // Check if sym is a nested function and we can declare it as internal.
    //
    // Nested naked functions and the implicitly generated __require/__ensure
    // functions for in/out contracts cannot be internalized. The reason
    // for the latter is that contract functions, despite being nested, can be
    // referenced from other D modules e.g. in the case of contracts on
    // interface methods (where __require/__ensure are emitted to the module
    // where the interface is declared, but an actual interface implementation
    // can be in a completely different place).
    FuncDeclaration* fd = sym->isFuncDeclaration();
    if (!fd || (!fd->naked && fd->ident != Id::require && fd->ident != Id::ensure))
    {
        // Any symbol nested in a function that cannot be inlined can't be
        // referenced directly from outside that function, so we can give
        // such symbols internal linkage. This holds even if nested indirectly,
        // such as member functions of aggregates nested in functions.
        //
        // Note: This must be checked after things like template member-ness or
        // symbols nested in templates would get duplicated for each module,
        // breaking things like
        // ---
        // int counter(T)() { static int i; return i++; }"
        // ---
        // if instances get emitted in multiple object files because they'd use
        // different instances of 'i'.
        // TODO: Check if we are giving away too much inlining potential due to
        // canInline being overly conservative here.
        for (Dsymbol* parent = sym->parent; parent ; parent = parent->parent)
        {
            FuncDeclaration *parentFd = parent->isFuncDeclaration();
            if (parentFd && !parentFd->canInline(parentFd->needThis(), false, false))
            {
                // We also cannot internalize nested functions which are
                // leaked to the outside via a templated return type, because
                // that type will also be codegen'd in any caller modules (see
                // GitHub issue #131).
                // Since we can't easily determine if this is really the case
                // here, just don't internalize it if the parent returns a
                // template at all, to be safe.
                TypeFunction* tf = static_cast<TypeFunction*>(parentFd->type);
                if (!DtoIsTemplateInstance(tf->next->toDsymbol(parentFd->scope)))
                    return llvm::GlobalValue::InternalLinkage;
            }
        }
    }
#endif
    // default to external linkage
    return llvm::GlobalValue::ExternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym)
{
    if (DtoIsTemplateInstance(sym))
        return templateLinkage;
    return llvm::GlobalValue::InternalLinkage;
}

llvm::GlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym)
{
    if (DtoIsTemplateInstance(sym))
        return templateLinkage;
    return llvm::GlobalValue::ExternalLinkage;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLIntegerType* DtoSize_t()
{
    // the type of size_t does not change once set
    static LLIntegerType* t = NULL;
    if (t == NULL)
        t = (global.params.is64bit) ? LLType::getInt64Ty(gIR->context()) : LLType::getInt32Ty(gIR->context());
    return t;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP1(LLValue* ptr, LLValue* i0, const char* var, llvm::BasicBlock* bb)
{
    return llvm::GetElementPtrInst::Create(ptr, i0, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var, llvm::BasicBlock* bb)
{
    LLValue* v[] = { i0, i1 };
    return llvm::GetElementPtrInst::Create(ptr, v, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi1(LLValue* ptr, unsigned i, const char* var, llvm::BasicBlock* bb)
{
    return llvm::GetElementPtrInst::Create(ptr, DtoConstUint(i), var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var, llvm::BasicBlock* bb)
{
    LLValue* v[] = { DtoConstUint(i0), DtoConstUint(i1) };
    return llvm::GetElementPtrInst::Create(ptr, v, var, bb ? bb : gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoGEPi(LLConstant* ptr, unsigned i0, unsigned i1)
{
    LLValue* v[] = { DtoConstUint(i0), DtoConstUint(i1) };
    return llvm::ConstantExpr::getGetElementPtr(ptr, v, true);
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
    LLFunction* fn = gIR->module->getFunction("memcmp");
    if (!fn)
    {
        LLType* Tys[] = { VoidPtrTy, VoidPtrTy, DtoSize_t() };
        LLFunctionType* fty = LLFunctionType::get(LLType::getInt32Ty(gIR->context()),
                                                  Tys, false);
        fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, "memcmp", gIR->module);
    }

    lhs = DtoBitCast(lhs, VoidPtrTy);
    rhs = DtoBitCast(rhs, VoidPtrTy);

    return gIR->ir->CreateCall3(fn, lhs, rhs, nbytes);
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
    LLConstant* init = llvm::ConstantDataArray::getString(gIR->context(), s, true);
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
        *gIR->module, init->getType(), true, llvm::GlobalValue::InternalLinkage, init, ".str");
    gvar->setUnnamedAddr(true);
    LLConstant* idxs[] = { DtoConstUint(0), DtoConstUint(0) };
    return DtoConstSlice(
        DtoConstSize_t(s.size()),
        llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true),
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


void DtoStore(LLValue* src, LLValue* dst)
{
    assert(src->getType() != llvm::Type::getInt1Ty(gIR->context()) &&
        "Should store bools as i8 instead of i1.");
    gIR->ir->CreateStore(src,dst);
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

LLStructType* DtoInterfaceInfoType()
{
    if (gIR->interfaceInfoType)
        return gIR->interfaceInfoType;

    // build interface info type
    LLSmallVector<LLType*, 3> types;
    // ClassInfo classinfo
    ClassDeclaration* cd2 = Type::typeinfoclass;
    DtoResolveClass(cd2);
    types.push_back(DtoType(cd2->type));
    // void*[] vtbl
    LLSmallVector<LLType*, 2> vtbltypes;
    vtbltypes.push_back(DtoSize_t());
    LLType* byteptrptrty = getPtrToType(getPtrToType(LLType::getInt8Ty(gIR->context())));
    vtbltypes.push_back(byteptrptrty);
    types.push_back(LLStructType::get(gIR->context(), vtbltypes));
    // int offset
    types.push_back(LLType::getInt32Ty(gIR->context()));
    // create type
    gIR->interfaceInfoType = LLStructType::get(gIR->context(), types);

    return gIR->interfaceInfoType;
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

LLValue* DtoAggrPairSwap(LLValue* aggr)
{
    Logger::println("swapping aggr pair");
    LLValue* r = gIR->ir->CreateExtractValue(aggr, 0);
    LLValue* i = gIR->ir->CreateExtractValue(aggr, 1);
    return DtoAggrPair(i, r, "swapped");
}
