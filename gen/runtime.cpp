//===-- runtime.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/runtime.h"
#include "aggregate.h"
#include "dsymbol.h"
#include "lexer.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "root.h"
#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irtype.h"
#include "ir/irtypefunction.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Module.h"
#include "llvm/IR/Attributes.h"
#else
#include "llvm/Module.h"
#include "llvm/Attributes.h"
#endif

#include <algorithm>

#if LDC_LLVM_VER < 302
using namespace llvm::Attribute;
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool> nogc("nogc",
    llvm::cl::desc("Do not allow code that generates implicit garbage collector calls"),
    llvm::cl::ZeroOrMore);

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::Module* M = NULL;

static void LLVM_D_BuildRuntimeModule();

//////////////////////////////////////////////////////////////////////////////////////////////////

static void checkForImplicitGCCall(const Loc &loc, const char *name)
{
    if (nogc)
    {
        static const std::string GCNAMES[] =
        {
            "_aaDelX",
            "_aaGetX",
            "_aaKeys",
            "_aaRehash",
            "_aaValues",
            "_adDupT",
            "_d_allocmemory",
            "_d_allocmemoryT",
            "_d_array_cast_len",
            "_d_array_slice_copy",
            "_d_arrayappendT",
            "_d_arrayappendcTX",
            "_d_arrayappendcd",
            "_d_arrayappendwd",
            "_d_arraycatT",
            "_d_arraycatnT",
            "_d_arraysetlengthT",
            "_d_arraysetlengthiT",
            "_d_assocarrayliteralTX",
            "_d_callfinalizer",
            "_d_delarray_t",
            "_d_delclass",
            "_d_delinterface",
            "_d_delmemory",
            "_d_newarrayT",
            "_d_newarrayiT",
            "_d_newarraymT",
            "_d_newarraymiT",
            "_d_newarrayU",
            "_d_newclass",
        };

        if (binary_search(&GCNAMES[0], &GCNAMES[sizeof(GCNAMES) / sizeof(std::string)], name))
        {
            error(loc, "No implicit garbage collector calls allowed with -nogc option enabled: %s", name);
            fatal();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

bool LLVM_D_InitRuntime()
{
    Logger::println("*** Initializing D runtime declarations ***");
    LOG_SCOPE;

    if (!M)
    {
        LLVM_D_BuildRuntimeModule();
    }

    return true;
}

void LLVM_D_FreeRuntime()
{
    if (M) {
        Logger::println("*** Freeing D runtime declarations ***");
        delete M;
        M = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_D_GetRuntimeFunction(const Loc &loc, llvm::Module* target, const char* name)
{
    checkForImplicitGCCall(loc, name);

    if (!M) {
        LLVM_D_InitRuntime();
    }

    LLFunction* fn = target->getFunction(name);
    if (fn)
        return fn;

    fn = M->getFunction(name);

    std::cout << name << std::endl;

    assert(fn && "Runtime function not found.");

    LLFunctionType* fnty = fn->getFunctionType();
    LLFunction* resfn = llvm::cast<llvm::Function>(target->getOrInsertFunction(name, fnty));
    resfn->setAttributes(fn->getAttributes());
    resfn->setCallingConv(fn->getCallingConv());
    return resfn;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(Loc& loc, llvm::Module* target, const char* name)
{
    LLGlobalVariable* gv = target->getNamedGlobal(name);
    if (gv) {
        return gv;
    }

    checkForImplicitGCCall(loc, name);

    if (!M) {
        LLVM_D_InitRuntime();
    }

    LLGlobalVariable* g = M->getNamedGlobal(name);
    if (!g) {
        error(loc, "Runtime global '%s' was not found", name);
        fatal();
        //return NULL;
    }

    LLPointerType* t = g->getType();
    return getOrCreateGlobal(loc, *target, t->getElementType(), g->isConstant(),
                             g->getLinkage(), NULL, g->getName());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLType* rt_ptr(LLType* t)
{
    return getPtrToType(t);
}

static LLType* rt_array(LLType* elemty)
{
    LLType *types[] = { DtoSize_t(), rt_ptr(elemty) };
    return LLStructType::get(gIR->context(), types, false);
}

static LLType* rt_dg1()
{
    LLType *types1[] = { rt_ptr(LLType::getInt8Ty(gIR->context())),
                         rt_ptr(LLType::getInt8Ty(gIR->context())) };
    LLFunctionType* fty = LLFunctionType::get(LLType::getInt32Ty(gIR->context()), types1, false);

    LLType *types[] = {
        rt_ptr(LLType::getInt8Ty(gIR->context())),
        rt_ptr(fty)
    };
    return LLStructType::get(gIR->context(), types, false);
}

static LLType* rt_dg2()
{
    LLType *Int8PtrTy = rt_ptr(LLType::getInt8Ty(gIR->context()));
    LLType *types1[] = { Int8PtrTy, Int8PtrTy, Int8PtrTy };
    LLFunctionType* fty = LLFunctionType::get(LLType::getInt32Ty(gIR->context()), types1, false);

    LLType *types[] = {
        rt_ptr(LLType::getInt8Ty(gIR->context())),
        rt_ptr(fty)
    };
    return LLStructType::get(gIR->context(), types, false);
}

template<typename DECL>
static void ensureDecl(DECL *decl, const char *msg)
{
    if (!decl || !decl->type)
    {
        Logger::println("Missing class declaration: %s\n", msg);
        error(Loc(), "Missing class declaration: %s", msg);
        errorSupplemental(Loc(), "Please check that object.di is included and valid");
        fatal();
    }
}

static LLFunction *LLVM_D_BuildFunctionFwdDecl(Type *returntype,
                                               llvm::StringRef fname,
                                               std::vector<LLType *> params,
                                               bool isVarArg,
                                               llvm::Module *M,
                                               LINK linkage = LINKc)
{
    LLType *rt = DtoType(returntype);
    bool sret = gABI->returnInArg(returntype, linkage);
    LLFunctionType *fty; // Initialized below
    if (sret)
    {
        params.insert(params.begin(), rt->getPointerTo());
        LLType* voidTy = DtoType(Type::tvoid);
        fty = LLFunctionType::get(voidTy, params, isVarArg);
    }
    else if (!params.empty())
    {
        fty = LLFunctionType::get(rt, params, isVarArg);
    }
    else
    {
        fty = LLFunctionType::get(rt, isVarArg);
    }

    LLFunction *fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, fname, M);
    if (sret) {
        fn->addAttribute(1, LDC_ATTRIBUTE(StructRet));
        fn->addAttribute(1, LDC_ATTRIBUTE(NoAlias));
    }
    fn->setCallingConv(gABI->callingConv(linkage));
    return fn;
}

static void LLVM_D_BuildRuntimeModule()
{
    Logger::println("building runtime module");
    M = new llvm::Module("ldc internal runtime", gIR->context());

    LLType* voidTy = LLType::getVoidTy(gIR->context());
    LLType* boolTy = LLType::getInt1Ty(gIR->context());
    LLType* byteTy = LLType::getInt8Ty(gIR->context());
    LLType* intTy = LLType::getInt32Ty(gIR->context());
    LLType* longTy = LLType::getInt64Ty(gIR->context());
    LLType* sizeTy = DtoSize_t();

    LLType* voidPtrTy = rt_ptr(byteTy);
    LLType* voidArrayTy = rt_array(byteTy);
    LLType* voidArrayPtrTy = getPtrToType(voidArrayTy);
    LLType* stringTy = DtoType(Type::tchar->arrayOf());
    LLType* wstringTy = DtoType(Type::twchar->arrayOf());
    LLType* dstringTy = DtoType(Type::tdchar->arrayOf());

    // Ensure that the declarations exist before creating llvm types for them.
    ensureDecl(ClassDeclaration::object, "Object");
    ensureDecl(Type::typeinfoclass, "TypeInfo_Class");
    ensureDecl(Type::dtypeinfo, "DTypeInfo");
    ensureDecl(Type::typeinfoassociativearray, "TypeInfo_AssociativeArray");
    ensureDecl(Module::moduleinfo, "ModuleInfo");

    LLType* objectTy = DtoType(ClassDeclaration::object->type);
    LLType* classInfoTy = DtoType(Type::typeinfoclass->type);
    LLType* typeInfoTy = DtoType(Type::dtypeinfo->type);
    LLType* aaTypeInfoTy = DtoType(Type::typeinfoassociativearray->type);
    LLType* moduleInfoPtrTy = getPtrToType(DtoType(Module::moduleinfo->type));
    LLType* aaTy = rt_ptr(LLStructType::get(gIR->context()));

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Construct some attribute lists used below (possibly multiple times)
#if LDC_LLVM_VER >= 303
    llvm::AttributeSet
        NoAttrs,
        Attr_NoAlias
            = NoAttrs.addAttribute(gIR->context(), 0, llvm::Attribute::NoAlias),
        Attr_NoUnwind
            = NoAttrs.addAttribute(gIR->context(), ~0U, llvm::Attribute::NoUnwind),
        Attr_ReadOnly
            = NoAttrs.addAttribute(gIR->context(), ~0U, llvm::Attribute::ReadOnly),
        Attr_ReadOnly_NoUnwind
            = Attr_ReadOnly.addAttribute(gIR->context(), ~0U, llvm::Attribute::NoUnwind),
        Attr_ReadOnly_1_NoCapture
            = Attr_ReadOnly.addAttribute(gIR->context(), 1, llvm::Attribute::NoCapture),
        Attr_ReadOnly_1_3_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttribute(gIR->context(), 3, llvm::Attribute::NoCapture),
        Attr_ReadOnly_NoUnwind_1_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttribute(gIR->context(), ~0U, llvm::Attribute::NoUnwind),
        Attr_ReadNone
            = NoAttrs.addAttribute(gIR->context(), ~0U, llvm::Attribute::ReadNone),
        Attr_1_NoCapture
            = NoAttrs.addAttribute(gIR->context(), 1, llvm::Attribute::NoCapture),
        Attr_NoAlias_1_NoCapture
            = Attr_1_NoCapture.addAttribute(gIR->context(), 0, llvm::Attribute::NoAlias),
        Attr_1_2_NoCapture
            = Attr_1_NoCapture.addAttribute(gIR->context(), 2, llvm::Attribute::NoCapture),
        Attr_1_3_NoCapture
            = Attr_1_NoCapture.addAttribute(gIR->context(), 3, llvm::Attribute::NoCapture),
        Attr_1_4_NoCapture
            = Attr_1_NoCapture.addAttribute(gIR->context(), 4, llvm::Attribute::NoCapture);
#elif LDC_LLVM_VER == 302
    llvm::AttrListPtr
        NoAttrs,
        Attr_NoAlias
            = NoAttrs.addAttr(gIR->context(), 0, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoAlias))),
        Attr_NoUnwind
            = NoAttrs.addAttr(gIR->context(), ~0U, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoUnwind))),
        Attr_ReadOnly
            = NoAttrs.addAttr(gIR->context(), ~0U, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::ReadOnly))),
        Attr_ReadOnly_NoUnwind
            = Attr_ReadOnly.addAttr(gIR->context(), ~0U, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoUnwind))),
        Attr_ReadOnly_1_NoCapture
            = Attr_ReadOnly.addAttr(gIR->context(), 1, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture))),
        Attr_ReadOnly_1_3_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(gIR->context(), 3, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture))),
        Attr_ReadOnly_NoUnwind_1_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(gIR->context(), ~0U, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoUnwind))),
        Attr_ReadNone
            = NoAttrs.addAttr(gIR->context(), ~0U, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::ReadNone))),
        Attr_1_NoCapture
            = NoAttrs.addAttr(gIR->context(), 1, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture))),
        Attr_NoAlias_1_NoCapture
            = Attr_1_NoCapture.addAttr(gIR->context(), 0, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoAlias))),
        Attr_1_2_NoCapture
            = Attr_1_NoCapture.addAttr(gIR->context(), 2, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture))),
        Attr_1_3_NoCapture
            = Attr_1_NoCapture.addAttr(gIR->context(), 3, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture))),
        Attr_1_4_NoCapture
            = Attr_1_NoCapture.addAttr(gIR->context(), 4, llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoCapture)));
#else
    llvm::AttrListPtr
        NoAttrs,
        Attr_NoAlias
            = NoAttrs.addAttr(0, NoAlias),
        Attr_NoUnwind
            = NoAttrs.addAttr(~0U, NoUnwind),
        Attr_ReadOnly
            = NoAttrs.addAttr(~0U, ReadOnly),
        Attr_ReadOnly_NoUnwind
            = Attr_ReadOnly.addAttr(~0U, NoUnwind),
        Attr_ReadOnly_1_NoCapture
            = Attr_ReadOnly.addAttr(1, NoCapture),
        Attr_ReadOnly_1_3_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(3, NoCapture),
        Attr_ReadOnly_NoUnwind_1_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(~0U, NoUnwind),
        Attr_ReadNone
            = NoAttrs.addAttr(~0U, ReadNone),
        Attr_1_NoCapture
            = NoAttrs.addAttr(1, NoCapture),
        Attr_NoAlias_1_NoCapture
            = Attr_1_NoCapture.addAttr(0, NoAlias),
        Attr_1_2_NoCapture
            = Attr_1_NoCapture.addAttr(2, NoCapture),
        Attr_1_3_NoCapture
            = Attr_1_NoCapture.addAttr(3, NoCapture),
        Attr_1_4_NoCapture
            = Attr_1_NoCapture.addAttr(4, NoCapture);
#endif

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_assert(string file, uint line)
    // void _d_arraybounds(string file, uint line)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_assert",      { stringTy, intTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_arraybounds", { stringTy, intTy }, false, M);

    // void _d_switch_error(ModuleInfo* m, uint line)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_switch_error", { moduleInfoPtrTy, intTy }, false, M);

    // void _d_assert_msg(string msg, string file, uint line)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_assert_msg", { stringTy, stringTy, intTy }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////


    // void* _d_allocmemory(size_t sz)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_d_allocmemory",  { sizeTy }, false, M)
        ->setAttributes(Attr_NoAlias);

    // void* _d_allocmemoryT(TypeInfo ti)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_d_allocmemoryT",  { typeInfoTy }, false, M)
        ->setAttributes(Attr_NoAlias);

    // void[] _d_newarrayT(TypeInfo ti, size_t length)
    // void[] _d_newarrayiT(TypeInfo ti, size_t length)
    // void[] _d_newarrayU(TypeInfo ti, size_t length)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_newarrayT",  { typeInfoTy, sizeTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_newarrayiT", { typeInfoTy, sizeTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_newarrayU",  { typeInfoTy, sizeTy }, false, M);

    // void[] _d_newarraymT(TypeInfo ti, size_t length, size_t* dims)
    // void[] _d_newarraymiT(TypeInfo ti, size_t length, size_t* dims)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_newarraymT",  { typeInfoTy, sizeTy }, true, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_newarraymiT", { typeInfoTy, sizeTy }, true, M);

    // void[] _d_arraysetlengthT(TypeInfo ti, size_t newlength, void[] *array)
    // void[] _d_arraysetlengthiT(TypeInfo ti, size_t newlength, void[] *array)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arraysetlengthT", 
        { typeInfoTy, sizeTy, voidArrayPtrTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arraysetlengthiT", 
        { typeInfoTy, sizeTy, voidArrayPtrTy }, false, M);

    // byte[] _d_arrayappendcTX(TypeInfo ti, ref byte[] px, size_t n)
    LLVM_D_BuildFunctionFwdDecl(Type::tint8->arrayOf(), "_d_arrayappendcTX", 
        { typeInfoTy, voidArrayPtrTy, sizeTy }, false, M);

    // void[] _d_arrayappendT(TypeInfo ti, byte[]* px, byte[] y)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arrayappendT", 
        { typeInfoTy, voidArrayPtrTy, voidArrayTy }, false, M);

    // void[] _d_arrayappendcd(ref char[] x, dchar c)
    // void[] _d_arrayappendwd(ref wchar[] x, dchar c)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arrayappendcd", 
        { getPtrToType(stringTy), intTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arrayappendwd", 
        { getPtrToType(wstringTy), intTy }, false, M);

    // byte[] _d_arraycatT(TypeInfo ti, byte[] x, byte[] y)
    LLVM_D_BuildFunctionFwdDecl(Type::tint8->arrayOf(), "_d_arraycatT", 
        { typeInfoTy, voidArrayTy, voidArrayTy }, false, M);

    // byte[] _d_arraycatnT(TypeInfo ti, uint n, ...)
    LLVM_D_BuildFunctionFwdDecl(Type::tint8->arrayOf(), "_d_arraycatnT", { typeInfoTy }, true, M);

    // Object _d_newclass(const ClassInfo ci)
    LLVM_D_BuildFunctionFwdDecl(ClassDeclaration::object->type, "_d_newclass",
        { classInfoTy }, false, M)
        ->setAttributes(Attr_NoAlias);

    // void _d_delarray_t(Array *p, TypeInfo ti)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_delarray_t", { voidArrayPtrTy, typeInfoTy }, false, M);

    // void _d_delmemory(void **p)
    // void _d_delinterface(void **p)
    // void _d_callfinalizer(void *p)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_delmemory",     { voidPtrTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_delinterface",  { voidPtrTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_callfinalizer", { voidPtrTy }, false, M);

    // void _d_delclass(Object* p)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_delclass", { rt_ptr(objectTy) }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // array slice copy when assertions are on!
    // void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t srclen)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_array_slice_copy",
        { voidPtrTy, sizeTy, voidPtrTy, sizeTy }, false, M)
        ->setAttributes(Attr_1_3_NoCapture);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _aApplycd1(char[] aa, dg_t dg)
    #define STR_APPLY1(TY,a,b) \
    { \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, a, { TY, rt_dg1() }, false, M); \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, b, { TY, rt_dg1() }, false, M); \
    }
    STR_APPLY1(stringTy, "_aApplycw1", "_aApplycd1")
    STR_APPLY1(wstringTy, "_aApplywc1", "_aApplywd1")
    STR_APPLY1(dstringTy, "_aApplydc1", "_aApplydw1")
    #undef STR_APPLY

    // int _aApplycd2(char[] aa, dg2_t dg)
    #define STR_APPLY2(TY,a,b) \
    { \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, a, { TY, rt_dg2() }, false, M); \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, b, { TY, rt_dg2() }, false, M); \
    }
    STR_APPLY2(stringTy, "_aApplycw2", "_aApplycd2")
    STR_APPLY2(wstringTy, "_aApplywc2", "_aApplywd2")
    STR_APPLY2(dstringTy, "_aApplydc2", "_aApplydw2")
    #undef STR_APPLY2

    #define STR_APPLY_R1(TY,a,b) \
    { \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, a, { TY, rt_dg1() }, false, M); \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, b, { TY, rt_dg1() }, false, M); \
    }
    STR_APPLY_R1(stringTy, "_aApplyRcw1", "_aApplyRcd1")
    STR_APPLY_R1(wstringTy, "_aApplyRwc1", "_aApplyRwd1")
    STR_APPLY_R1(dstringTy, "_aApplyRdc1", "_aApplyRdw1")
    #undef STR_APPLY

    #define STR_APPLY_R2(TY,a,b) \
    { \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, a, { TY, rt_dg2() }, false, M); \
        LLVM_D_BuildFunctionFwdDecl(Type::tint32, b, { TY, rt_dg2() }, false, M); \
    }
    STR_APPLY_R2(stringTy, "_aApplyRcw2", "_aApplyRcd2")
    STR_APPLY_R2(wstringTy, "_aApplyRwc2", "_aApplyRwd2")
    STR_APPLY_R2(dstringTy, "_aApplyRdc2", "_aApplyRdw2")
    #undef STR_APPLY2

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // fixes the length for dynamic array casts
    // size_t _d_array_cast_len(size_t len, size_t elemsz, size_t newelemsz)
    LLVM_D_BuildFunctionFwdDecl(Type::tsize_t, "_d_array_cast_len", { sizeTy, sizeTy, sizeTy }, false, M)
        ->setAttributes(Attr_ReadNone);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void[] _d_arrayassign(TypeInfo ti, void[] from, void[] to)
    // void[] _d_arrayctor(TypeInfo ti, void[] from, void[] to)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arrayassign",
                                { typeInfoTy, voidArrayTy, voidArrayTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_d_arrayctor",
                                { typeInfoTy, voidArrayTy, voidArrayTy }, false, M);

    // void* _d_arraysetassign(void* p, void* value, size_t count, TypeInfo ti)
    // void* _d_arraysetctor(void* p, void* value, size_t count, TypeInfo ti)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_d_arraysetassign",
                                { voidPtrTy, voidPtrTy, sizeTy, typeInfoTy }, false, M)
        ->setAttributes(Attr_NoAlias);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_d_arraysetctor",
                                { voidPtrTy, voidPtrTy, sizeTy, typeInfoTy }, false, M)
        ->setAttributes(Attr_NoAlias);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // cast to object
    // Object _d_toObject(void* p)
    LLVM_D_BuildFunctionFwdDecl(ClassDeclaration::object->type, "_d_toObject", { voidPtrTy }, false, M)
        ->setAttributes(Attr_ReadOnly_NoUnwind);

    // cast interface
    // Object _d_interface_cast(void* p, ClassInfo c)
    LLVM_D_BuildFunctionFwdDecl(ClassDeclaration::object->type, "_d_interface_cast",
                                { voidPtrTy, classInfoTy }, false, M)
        ->setAttributes(Attr_ReadOnly_NoUnwind);

    // dynamic cast
    // Object _d_dynamic_cast(Object o, ClassInfo c)
    LLVM_D_BuildFunctionFwdDecl(ClassDeclaration::object->type, "_d_dynamic_cast",
                                { objectTy, classInfoTy }, false, M)
        ->setAttributes(Attr_ReadOnly_NoUnwind);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // char[] _adReverseChar(char[] a)
    // char[] _adSortChar(char[] a)
    LLVM_D_BuildFunctionFwdDecl(Type::tchar->arrayOf(), "_adReverseChar", { stringTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tchar->arrayOf(), "_adSortChar",    { stringTy }, false, M);

    // wchar[] _adReverseWchar(wchar[] a)
    // wchar[] _adSortWchar(wchar[] a)
    LLVM_D_BuildFunctionFwdDecl(Type::twchar->arrayOf(), "_adReverseWchar", { wstringTy }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::twchar->arrayOf(), "_adSortWChar",    { wstringTy }, false, M);

    // void[] _adReverse(void[] a, size_t szelem)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_adReverse",
                                { rt_array(byteTy), sizeTy }, false, M)
        ->setAttributes(Attr_NoUnwind);

    // void[] _adDupT(TypeInfo ti, void[] a)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_adDupT",
                                { typeInfoTy, rt_array(byteTy) }, false, M);

    // int _adEq(void[] a1, void[] a2, TypeInfo ti)
    // int _adCmp(void[] a1, void[] a2, TypeInfo ti)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, _adEq,
                                { rt_array(byteTy), rt_array(byteTy), typeInfoTy }, false, M)
        ->setAttributes(Attr_ReadOnly);
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, _adCmp,
                                { rt_array(byteTy), rt_array(byteTy), typeInfoTy }, false, M)
        ->setAttributes(Attr_ReadOnly);

    // int _adCmpChar(void[] a1, void[] a2)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_adCmpChar",
                                {rt_array(byteTy), rt_array(byteTy)}, false,M)
        ->setAttributes(Attr_ReadOnly_NoUnwind);

    // void[] _adSort(void[] a, TypeInfo ti)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_adSort",
        { rt_array(byteTy), typeInfoTy }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // size_t _aaLen(AA aa)
    LLVM_D_BuildFunctionFwdDecl(Type::tsize_t, "_aaLen", { aaTy }, false, M)
        ->setAttributes(Attr_ReadOnly_NoUnwind_1_NoCapture);

    // void* _aaGetX(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_aaGetX",
                                { aaTy, typeInfoTy, sizeTy, voidPtrTy }, false, M)
        ->setAttributes(Attr_1_4_NoCapture);

    // void* _aaInX(AA aa, TypeInfo keyti, void* pkey)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_aaInX",
                                { aaTy, typeInfoTy, voidPtrTy }, false, M)
        ->setAttributes(Attr_ReadOnly_1_3_NoCapture);

    // bool _aaDelX(AA aa, TypeInfo keyti, void* pkey)
    LLVM_D_BuildFunctionFwdDecl(Type::tbool, "_aaDelX",
                                { aaTy, typeInfoTy, voidPtrTy }, false, M)
        ->setAttributes(Attr_1_3_NoCapture);

    // void[] _aaValues(AA aa, size_t keysize, size_t valuesize)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_aaValues",
                                { aaTy, sizeTy, sizeTy }, false, M)
        ->setAttributes(Attr_NoAlias_1_NoCapture);

    // void* _aaRehash(AA* paa, TypeInfo keyti)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_aaRehash", { aaTy, typeInfoTy }, false, M);

    // void[] _aaKeys(AA aa, size_t keysize)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid->arrayOf(), "_aaKeys",
                                { aaTy, sizeTy }, false, M)
        ->setAttributes(Attr_NoAlias_1_NoCapture);

    // int _aaApply(AA aa, size_t keysize, dg_t dg)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_aaApply",
                                { aaTy, sizeTy, rt_dg1() }, false, M)
        ->setAttributes(Attr_1_NoCapture);

    // int _aaApply2(AA aa, size_t keysize, dg2_t dg)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_aaApply2",
                                { aaTy, sizeTy, rt_dg2() }, false, M)
        ->setAttributes(Attr_1_NoCapture);

    // int _aaEqual(in TypeInfo tiRaw, in AA e1, in AA e2)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_aaEqual",
                                { typeInfoTy, aaTy, aaTy }, false, M)
        ->setAttributes(Attr_1_2_NoCapture);

    // BB* _d_assocarrayliteralTX(TypeInfo_AssociativeArray ti, void[] keys, void[] values)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoidptr, "_d_assocarrayliteralTX",
                                { aaTypeInfoTy, voidArrayTy, voidArrayTy }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _moduleCtor()
    // void _moduleDtor()
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_moduleCtor", { }, false, M);
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_moduleDtor", { }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_throw_exception(Object e)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_throw_exception", { objectTy }, false, M);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_switch_string(char[][] table, char[] ca)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_d_switch_string",
                                { rt_array(stringTy), stringTy }, false, M)
        ->setAttributes(Attr_ReadOnly);

    // int _d_switch_ustring(wchar[][] table, wchar[] ca)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_d_switch_ustring",
                                { rt_array(wstringTy), wstringTy }, false, M)
        ->setAttributes(Attr_ReadOnly);

    // int _d_switch_dstring(dchar[][] table, dchar[] ca)
    LLVM_D_BuildFunctionFwdDecl(Type::tint32, "_d_switch_dstring",
                                { rt_array(dstringTy), dstringTy }, false, M)
        ->setAttributes(Attr_ReadOnly);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_eh_personality(...)
    {
        LLFunctionType* fty = NULL;
#if LDC_LLVM_VER >= 305
        if (global.params.targetTriple.isWindowsMSVCEnvironment())
        {
            // int _d_eh_personality(ptr ExceptionRecord, ptr EstablisherFrame, ptr ContextRecord, ptr DispatcherContext)
            LLType *types[] = { voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy };
            fty = llvm::FunctionType::get(intTy, types, false);
        }
        else
#endif
        if (global.params.targetTriple.getArch() == llvm::Triple::arm)
        {
            // int _d_eh_personality(int state, ptr ucb, ptr context)
            LLType *types[] = { intTy, voidPtrTy, voidPtrTy };
            fty = llvm::FunctionType::get(intTy, types, false);
        }
        else
        {
            // int _d_eh_personality(int ver, int actions, ulong eh_class, ptr eh_info, ptr context)
            LLType *types[] = { intTy, intTy, longTy, voidPtrTy, voidPtrTy };
            fty = llvm::FunctionType::get(intTy, types, false);
        }

        llvm::StringRef fname("_d_eh_personality");
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_eh_resume_unwind(ptr exc_struct)
    {
        llvm::StringRef fname("_d_eh_resume_unwind");
        LLType *types[] = { voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_eh_handle_collision(ptr exc_struct, ptr exc_struct)
    {
        llvm::StringRef fname("_d_eh_handle_collision");
        LLType *types[] = { voidPtrTy, voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void invariant._d_invariant(Object o)
    {
        // KLUDGE: _d_invariant is actually extern(D) in the upstream runtime, possibly
        // for more efficient parameter passing on x86. This complicates our code here
        // quite a bit, though.
        Parameters* params = new Parameters();
        params->push(new Parameter(STCin, ClassDeclaration::object->type, NULL, NULL));
        TypeFunction* dty = new TypeFunction(params, Type::tvoid, 0, LINKd);
        llvm::Function* fn = llvm::Function::Create(
            llvm::cast<llvm::FunctionType>(DtoType(dty)),
            llvm::GlobalValue::ExternalLinkage,
            gABI->mangleForLLVM("_D9invariant12_d_invariantFC6ObjectZv", LINKd),
            M
        );
        assert(dty->ctype);
        IrFuncTy &irFty = dty->ctype->getIrFuncTy();
        gABI->rewriteFunctionType(dty, irFty);
#if LDC_LLVM_VER >= 303
        fn->addAttributes(1, llvm::AttributeSet::get(gIR->context(), 1, irFty.args[0]->attrs.attrs));
#elif LDC_LLVM_VER == 302
        fn->addAttribute(1, llvm::Attributes::get(gIR->context(), irFty.args[0]->attrs.attrs));
#else
        fn->addAttribute(1, irFty.args[0]->attrs.attrs);
#endif
        fn->setCallingConv(gABI->callingConv(LINKd));
    }

    // void _d_hidden_func(Object o)
    LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_hidden_func", { voidPtrTy }, false, M);

    // void _d_dso_registry(CompilerDSOData* data)
    if (global.params.isLinux) {
        llvm::StructType* dsoDataTy = llvm::StructType::get(
            sizeTy, // version
            getPtrToType(voidPtrTy), // slot
            getPtrToType(moduleInfoPtrTy), // _minfo_beg
            getPtrToType(moduleInfoPtrTy), // _minfo_end
            NULL
        );
        LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_dso_registry", { getPtrToType(dsoDataTy) }, false, M);
    }

    // void _d_cover_register2(string filename, size_t[] valid, uint[] data, ubyte minPercent)
    if (global.params.cov) {
        LLVM_D_BuildFunctionFwdDecl(Type::tvoid, "_d_cover_register2",
                                    { stringTy, rt_array(sizeTy), rt_array(intTy), byteTy }, false, M);
    }
}
