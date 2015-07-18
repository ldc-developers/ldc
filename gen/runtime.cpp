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
            "_d_arraycatnTX",
            "_d_arraysetlengthT",
            "_d_arraysetlengthiT",
            "_d_assocarrayliteralTX",
            "_d_callfinalizer",
            "_d_delarray_t",
            "_d_delclass",
            "_d_delstruct",
            "_d_delinterface",
            "_d_delmemory",
            "_d_newarrayT",
            "_d_newarrayiT",
            "_d_newarraymTX",
            "_d_newarraymiTX",
            "_d_newarrayU",
            "_d_newclass",
            "_d_newitemT",
            "_d_newitemiT",
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

llvm::Function* LLVM_D_GetRuntimeFunction(const Loc &loc, llvm::Module& target, const char* name)
{
    checkForImplicitGCCall(loc, name);

    if (!M) {
        LLVM_D_InitRuntime();
    }

    LLFunction* fn = target.getFunction(name);
    if (fn)
        return fn;

    fn = M->getFunction(name);
    assert(fn && "Runtime function not found.");

    LLFunctionType* fnty = fn->getFunctionType();
    LLFunction* resfn = llvm::cast<llvm::Function>(target.getOrInsertFunction(name, fnty));
    resfn->setAttributes(fn->getAttributes());
    resfn->setCallingConv(fn->getCallingConv());
    return resfn;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(Loc& loc, llvm::Module& target, const char* name)
{
    LLGlobalVariable* gv = target.getNamedGlobal(name);
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
    return getOrCreateGlobal(loc, target, t->getElementType(), g->isConstant(),
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
    LLType* voidArrayPtrTy = rt_ptr(voidArrayTy);
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
    LLType* moduleInfoPtrTy = rt_ptr(DtoType(Module::moduleinfo->type));
    LLType* aaTy = voidPtrTy;

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
    {
        llvm::StringRef fname ("_d_assert");
        llvm::StringRef fname2("_d_arraybounds");
        LLType *types[] = { stringTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_assert_msg(string msg, string file, uint line)
    {
        llvm::StringRef fname("_d_assert_msg");
        LLType *types[] = { stringTy, stringTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_assertm(immutable(ModuleInfo)* m, uint line)
    // void _d_array_bounds(immutable(ModuleInfo)* m, uint line)
    {
        llvm::StringRef fname ("_d_assertm");
        llvm::StringRef fname2("_d_array_bounds");
        LLType *types[] = { moduleInfoPtrTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_switch_error(immutable(ModuleInfo)* m, uint line)
    {
        llvm::StringRef fname("_d_switch_error");
        LLType *types[] = { moduleInfoPtrTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////


    // void* _d_allocmemory(size_t sz)
    {
        llvm::StringRef fname("_d_allocmemory");
        LLType *types[] = { sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
                ->setAttributes(Attr_NoAlias);
    }

    // void* _d_allocmemoryT(TypeInfo ti)
    {
        llvm::StringRef fname("_d_allocmemoryT");
        LLType *types[] = { typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
    }
    // void[] _d_newarrayT (const TypeInfo ti, size_t length)
    // void[] _d_newarrayiT(const TypeInfo ti, size_t length)
    // void[] _d_newarrayU (const TypeInfo ti, size_t length)
    {
        llvm::StringRef fname ("_d_newarrayT");
        llvm::StringRef fname2("_d_newarrayiT");
        llvm::StringRef fname3("_d_newarrayU");
        LLType *types[] = { typeInfoTy, sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }
    // void[] _d_newarraymTX (const TypeInfo ti, size_t[] dims)
    // void[] _d_newarraymiTX(const TypeInfo ti, size_t[] dims)
    {
        llvm::StringRef fname ("_d_newarraymTX");
        llvm::StringRef fname2("_d_newarraymiTX");
        LLType *types[] = { typeInfoTy, rt_array(sizeTy) };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void[] _d_arraysetlengthT (const TypeInfo ti, size_t newlength, void[]* p)
    // void[] _d_arraysetlengthiT(const TypeInfo ti, size_t newlength, void[]* p)
    {
        llvm::StringRef fname ("_d_arraysetlengthT");
        llvm::StringRef fname2("_d_arraysetlengthiT");
        LLType *types[] = { typeInfoTy, sizeTy, voidArrayPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // byte[] _d_arrayappendcTX(const TypeInfo ti, ref byte[] px, size_t n)
    {
        llvm::StringRef fname("_d_arrayappendcTX");
        LLType *types[] = { typeInfoTy, voidArrayPtrTy, sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendT(const TypeInfo ti, ref byte[] x, byte[] y)
    {
        llvm::StringRef fname("_d_arrayappendT");
        LLType *types[] = { typeInfoTy, voidArrayPtrTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendcd(ref byte[] x, dchar c)
    {
        llvm::StringRef fname("_d_arrayappendcd");
        LLType *types[] = { voidArrayPtrTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendwd(ref byte[] x, dchar c)
    {
        llvm::StringRef fname("_d_arrayappendwd");
        LLType *types[] = { voidArrayPtrTy, intTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // byte[] _d_arraycatT(const TypeInfo ti, byte[] x, byte[] y)
    {
        llvm::StringRef fname("_d_arraycatT");
        LLType *types[] = { typeInfoTy, voidArrayTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arraycatnTX(const TypeInfo ti, byte[][] arrs)
    {
        llvm::StringRef fname("_d_arraycatnTX");
        LLType* types[] = { typeInfoTy, rt_array(voidArrayTy) };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_newclass(const ClassInfo ci)
    {
        llvm::StringRef fname("_d_newclass");
        LLType *types[] = { classInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
    }

    // void* _d_newitemT (TypeInfo ti)
    // void* _d_newitemiT(TypeInfo ti)
    {
        llvm::StringRef fname ("_d_newitemT");
        llvm::StringRef fname2("_d_newitemiT");
        LLType *types[] = { typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_NoAlias);
    }

    // void _d_delarray_t(void[]* p, const TypeInfo_Struct ti)
    {
        llvm::StringRef fname("_d_delarray_t");
        LLType *types[] = { voidArrayPtrTy, DtoType(Type::typeinfostruct->type) };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_delmemory(void** p)
    // void _d_delinterface(void** p)
    {
        llvm::StringRef fname("_d_delmemory");
        llvm::StringRef fname2("_d_delinterface");
        LLType *types[] = { rt_ptr(voidPtrTy) };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_callfinalizer(void* p)
    {
        llvm::StringRef fname("_d_callfinalizer");
        LLType *types[] = { voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // D2: void _d_delclass(Object* p)
    {
        llvm::StringRef fname("_d_delclass");
        LLType *types[] = { rt_ptr(objectTy) };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_delstruct(void** p, TypeInfo_Struct inf)
    {
        llvm::StringRef fname("_d_delstruct");
        LLType *types[] = { rt_ptr(voidPtrTy), DtoType(Type::typeinfostruct->type) };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // array slice copy when assertions are on!
    // void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t srclen)
    {
        llvm::StringRef fname("_d_array_slice_copy");
        LLType *types[] = { voidPtrTy, sizeTy, voidPtrTy, sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_3_NoCapture);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _aApplycd1(in char[] aa, dg_t dg)
    // int _aApplyRcd1(in char[] aa, dg_t dg)
    #define STR_APPLY1(TY,a,b) \
    { \
        const std::string prefix = "_aApply"; \
        std::string fname1 = prefix + a + '1', \
                    fname2 = prefix + b + '1', \
                    fname3 = prefix + 'R' + a + '1', \
                    fname4 = prefix + 'R' + b + '1'; \
        LLType *types[] = { TY, rt_dg1() }; \
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname1, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname4, M); \
    }
    STR_APPLY1(stringTy,  "cw", "cd")
    STR_APPLY1(wstringTy, "wc", "wd")
    STR_APPLY1(dstringTy, "dc", "dw")
    #undef STR_APPLY1

    // int _aApplycd2(in char[] aa, dg2_t dg)
    // int _aApplyRcd2(in char[] aa, dg2_t dg)
    #define STR_APPLY2(TY,a,b) \
    { \
        const std::string prefix = "_aApply"; \
        std::string fname1 = prefix + a + '2', \
                    fname2 = prefix + b + '2', \
                    fname3 = prefix + 'R' + a + '2', \
                    fname4 = prefix + 'R' + b + '2'; \
        LLType *types[] = { TY, rt_dg2() }; \
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname1, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname4, M); \
    }
    STR_APPLY2(stringTy,  "cw", "cd")
    STR_APPLY2(wstringTy, "wc", "wd")
    STR_APPLY2(dstringTy, "dc", "dw")
    #undef STR_APPLY2

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // fixes the length for dynamic array casts
    // size_t _d_array_cast_len(size_t len, size_t elemsz, size_t newelemsz)
    {
        llvm::StringRef fname("_d_array_cast_len");
        LLType *types[] = { sizeTy, sizeTy, sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadNone);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void[] _d_arrayassign(TypeInfo ti, void[] from, void[] to)
    // void[] _d_arrayctor(TypeInfo ti, void[] from, void[] to)
    {
        llvm::StringRef fname("_d_arrayassign");
        llvm::StringRef fname2("_d_arrayctor");
        LLType *types[] = { typeInfoTy, voidArrayTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void* _d_arraysetassign(void* p, void* value, int count, TypeInfo ti)
    // void* _d_arraysetctor(void* p, void* value, int count, TypeInfo ti)
    {
        llvm::StringRef fname("_d_arraysetassign");
        llvm::StringRef fname2("_d_arraysetctor");
        LLType *types[] = { voidPtrTy, voidPtrTy, intTy, typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_NoAlias);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // cast to object
    // Object _d_toObject(void* p)
    {
        llvm::StringRef fname("_d_toObject");
        LLType *types[] = { voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // cast interface
    // void* _d_interface_cast(void* p, ClassInfo c)
    {
        llvm::StringRef fname("_d_interface_cast");
        LLType *types[] = { voidPtrTy, classInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // dynamic cast
    // void* _d_dynamic_cast(Object o, ClassInfo c)
    {
        llvm::StringRef fname("_d_dynamic_cast");
        LLType *types[] = { objectTy, classInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // char[] _adReverseChar(char[] a)
    // char[] _adSortChar(char[] a)
    {
        llvm::StringRef fname("_adReverseChar");
        llvm::StringRef fname2("_adSortChar");
        LLType *types[] = { stringTy };
        LLFunctionType* fty = llvm::FunctionType::get(stringTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // wchar[] _adReverseWchar(wchar[] a)
    // wchar[] _adSortWchar(wchar[] a)
    {
        llvm::StringRef fname("_adReverseWchar");
        llvm::StringRef fname2("_adSortWchar");
        LLType *types[] = { wstringTy };
        LLFunctionType* fty = llvm::FunctionType::get(wstringTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void[] _adReverse(void[] a, size_t szelem)
    {
        llvm::StringRef fname("_adReverse");
        LLType *types[] = { voidArrayTy, sizeTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoUnwind);
    }

    // void[] _adDupT(const TypeInfo ti, void[] a)
    {
        llvm::StringRef fname("_adDupT");
        LLType *types[] = { typeInfoTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _adEq(void[] a1, void[] a2, TypeInfo ti)
    // int _adCmp(void[] a1, void[] a2, TypeInfo ti)
    {
        llvm::StringRef fname(_adEq);
        llvm::StringRef fname2(_adCmp);
        LLType *types[] = { voidArrayTy, voidArrayTy, typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _adCmpChar(void[] a1, void[] a2)
    {
        llvm::StringRef fname("_adCmpChar");
        LLType *types[] = { voidArrayTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // void[] _adSort(void[] a, TypeInfo ti)
    {
        llvm::StringRef fname("_adSort");
        LLType *types[] = { voidArrayTy, typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // size_t _aaLen(in AA aa)
    {
        llvm::StringRef fname("_aaLen");
        LLType *types[] = { aaTy };
        LLFunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind_1_NoCapture);
    }

    // void* _aaGetX(AA* aa, const TypeInfo keyti, in size_t valuesize, in void* pkey)
    {
        llvm::StringRef fname("_aaGetX");
        LLType *types[] = { rt_ptr(aaTy), typeInfoTy, sizeTy, voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_4_NoCapture);
    }

    // inout(void)* _aaInX(inout AA aa, in TypeInfo keyti, in void* pkey)
    {
        llvm::StringRef fname("_aaInX");
        LLType *types[] = { aaTy, typeInfoTy, voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_1_3_NoCapture);
    }

    // bool _aaDelX(AA aa, in TypeInfo keyti, in void* pkey)
    {
        llvm::StringRef fname("_aaDelX");
        LLType *types[] = { aaTy, typeInfoTy, voidPtrTy };
        LLFunctionType* fty = llvm::FunctionType::get(boolTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_3_NoCapture);
    }

    // inout(void[]) _aaValues(inout AA aa, in size_t keysize, in size_t valuesize, const TypeInfo tiValueArray)
    {
        llvm::StringRef fname("_aaValues");
        LLType *types[] = { aaTy, sizeTy, sizeTy, typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias_1_NoCapture);
    }

    // void* _aaRehash(AA* paa, in TypeInfo keyti)
    {
        llvm::StringRef fname("_aaRehash");
        LLType *types[] = { rt_ptr(aaTy), typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // inout(void[]) _aaKeys(inout AA aa, in size_t keysize, const TypeInfo tiKeyArray)
    {
        llvm::StringRef fname("_aaKeys");
        LLType *types[] = { aaTy, sizeTy, typeInfoTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias_1_NoCapture);
    }

    // int _aaApply(AA aa, in size_t keysize, dg_t dg)
    {
        llvm::StringRef fname("_aaApply");
        LLType *types[] = { aaTy, sizeTy, rt_dg1() };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_NoCapture);
    }

    // int _aaApply2(AA aa, in size_t keysize, dg2_t dg)
    {
        llvm::StringRef fname("_aaApply2");
        LLType *types[] = { aaTy, sizeTy, rt_dg2() };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_NoCapture);
    }

    // int _aaEqual(in TypeInfo tiRaw, in AA e1, in AA e2)
    {
        llvm::StringRef fname("_aaEqual");
        LLType *types[] = { typeInfoTy, aaTy, aaTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_2_NoCapture);
    }
    // AA _d_assocarrayliteralTX(const TypeInfo_AssociativeArray ti, void[] keys, void[] values)
    {
        llvm::StringRef fname("_d_assocarrayliteralTX");
        LLType *types[] = { aaTypeInfoTy, voidArrayTy, voidArrayTy };
        LLFunctionType* fty = llvm::FunctionType::get(aaTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _moduleCtor()
    // void _moduleDtor()
    {
        llvm::StringRef fname("_moduleCtor");
        llvm::StringRef fname2("_moduleDtor");
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_throw_exception(Object e)
    {
        llvm::StringRef fname("_d_throw_exception");
        LLType *types[] = { objectTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_switch_string(char[][] table, char[] ca)
    {
        llvm::StringRef fname("_d_switch_string");
        LLType *types[] = { rt_array(stringTy), stringTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _d_switch_ustring(wchar[][] table, wchar[] ca)
    {
        llvm::StringRef fname("_d_switch_ustring");
        LLType *types[] = { rt_array(wstringTy), wstringTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _d_switch_dstring(dchar[][] table, dchar[] ca)
    {
        llvm::StringRef fname("_d_switch_dstring");
        LLType *types[] = { rt_array(dstringTy), dstringTy };
        LLFunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

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
        fn->setCallingConv(gABI->callingConv(fn->getFunctionType(), LINKd));
    }

    // void _d_hidden_func(Object o)
    {
        llvm::StringRef fname("_d_hidden_func");
        LLType *types[] = { objectTy };
        LLFunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_dso_registry(CompilerDSOData* data)
    if (global.params.isLinux) {
        llvm::StringRef fname("_d_dso_registry");

        llvm::StructType* dsoDataTy = llvm::StructType::get(
            sizeTy,                  // version
            rt_ptr(voidPtrTy),       // slot
            rt_ptr(moduleInfoPtrTy), // _minfo_beg
            rt_ptr(moduleInfoPtrTy), // _minfo_end
            NULL
        );

        llvm::Type* types[] = { rt_ptr(dsoDataTy) };
        llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // extern (C) void _d_cover_register2(string filename, size_t[] valid, uint[] data, ubyte minPercent)
    // as defined in druntime/rt/cover.d.
    if (global.params.cov) {
        llvm::StringRef fname("_d_cover_register2");

        LLType* params[] = {
            stringTy,
            rt_array(sizeTy),
            rt_array(intTy),
            byteTy
        };

        LLFunctionType* fty = LLFunctionType::get(voidTy, params, false);
        llvm::Function* fn = LLFunction::Create(fty, LLGlobalValue::ExternalLinkage, fname, M);
        fn->setCallingConv(gABI->callingConv(fn->getFunctionType(), LINKc));
    }
}
