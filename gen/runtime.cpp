#include "gen/llvm.h"
#include "llvm/Module.h"
#include "llvm/Attributes.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

#include "root.h"
#include "mars.h"
#include "lexer.h"
#include "dsymbol.h"
#include "mtype.h"
#include "aggregate.h"
#include "module.h"

#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "ir/irtype.h"

using namespace llvm::Attribute;

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool> noruntime("noruntime",
    llvm::cl::desc("Do not allow code that generates implicit runtime calls"),
    llvm::cl::ZeroOrMore);

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::Module* M = NULL;
static bool runtime_failed = false;

static void LLVM_D_BuildRuntimeModule();

//////////////////////////////////////////////////////////////////////////////////////////////////

bool LLVM_D_InitRuntime()
{
    Logger::println("*** Initializing D runtime declarations ***");
    LOG_SCOPE;

    if (!M)
        LLVM_D_BuildRuntimeModule();

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

llvm::Function* LLVM_D_GetRuntimeFunction(llvm::Module* target, const char* name)
{
    if (noruntime) {
        error("No implicit runtime calls allowed with -noruntime option enabled");
        fatal();
    }

    if (!M) {
        assert(!runtime_failed);
        LLVM_D_InitRuntime();
    }

    llvm::Function* fn = target->getFunction(name);
    if (fn)
        return fn;

    fn = M->getFunction(name);
    if (!fn) {
        printf("Runtime function '%s' was not found\n", name);
        assert(0);
        //return NULL;
    }

    const llvm::FunctionType* fnty = fn->getFunctionType();
    llvm::Function* resfn = llvm::cast<llvm::Function>(target->getOrInsertFunction(name, fnty));
    resfn->setAttributes(fn->getAttributes());
    return resfn;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name)
{
    llvm::GlobalVariable* gv = target->getNamedGlobal(name);
    if (gv) {
        return gv;
    }

    if (noruntime) {
        error("No implicit runtime calls allowed with -noruntime option enabled");
        fatal();
    }

    if (!M) {
        assert(!runtime_failed);
        LLVM_D_InitRuntime();
    }

    llvm::GlobalVariable* g = M->getNamedGlobal(name);
    if (!g) {
        error("Runtime global '%s' was not found", name);
        fatal();
        //return NULL;
    }

    const llvm::PointerType* t = g->getType();
    return new llvm::GlobalVariable(*target, t->getElementType(),g->isConstant(),g->getLinkage(),NULL,g->getName());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static const LLType* rt_ptr(const LLType* t)
{
    return getPtrToType(t);
}

static const LLType* rt_array(const LLType* elemty)
{
    return llvm::StructType::get(gIR->context(), DtoSize_t(), rt_ptr(elemty), NULL);
}

static const LLType* rt_dg1()
{
    std::vector<const LLType*> types;
    types.push_back(rt_ptr(LLType::getInt8Ty(gIR->context())));
    types.push_back(rt_ptr(LLType::getInt8Ty(gIR->context())));
    const llvm::FunctionType* fty = llvm::FunctionType::get(LLType::getInt32Ty(gIR->context()), types, false);
    return llvm::StructType::get(gIR->context(), rt_ptr(LLType::getInt8Ty(gIR->context())), rt_ptr(fty), NULL);
}

static const LLType* rt_dg2()
{
    std::vector<const LLType*> types;
    types.push_back(rt_ptr(LLType::getInt8Ty(gIR->context())));
    types.push_back(rt_ptr(LLType::getInt8Ty(gIR->context())));
    types.push_back(rt_ptr(LLType::getInt8Ty(gIR->context())));
    const llvm::FunctionType* fty = llvm::FunctionType::get(LLType::getInt32Ty(gIR->context()), types, false);
    return llvm::StructType::get(gIR->context(), rt_ptr(LLType::getInt8Ty(gIR->context())), rt_ptr(fty), NULL);
}

static void LLVM_D_BuildRuntimeModule()
{
    Logger::println("building module");
    M = new llvm::Module("ldc internal runtime", gIR->context());

    Logger::println("building basic types");
    const LLType* voidTy = LLType::getVoidTy(gIR->context());
    const LLType* boolTy = LLType::getInt1Ty(gIR->context());
    const LLType* byteTy = LLType::getInt8Ty(gIR->context());
    const LLType* shortTy = LLType::getInt16Ty(gIR->context());
    const LLType* intTy = LLType::getInt32Ty(gIR->context());
    const LLType* longTy = LLType::getInt64Ty(gIR->context());
    const LLType* sizeTy = DtoSize_t();

    Logger::println("building float types");
    const LLType* floatTy = LLType::getFloatTy(gIR->context());
    const LLType* doubleTy = LLType::getDoubleTy(gIR->context());
    const LLType* realTy;
    if ((global.params.cpu == ARCHx86) || (global.params.cpu == ARCHx86_64))
        realTy = LLType::getX86_FP80Ty(gIR->context());
    else
        realTy = LLType::getDoubleTy(gIR->context());

    const LLType* cfloatTy = llvm::StructType::get(gIR->context(), floatTy, floatTy, NULL);
    const LLType* cdoubleTy = llvm::StructType::get(gIR->context(), doubleTy, doubleTy, NULL);
    const LLType* crealTy = llvm::StructType::get(gIR->context(), realTy, realTy, NULL);

    Logger::println("building aggr types");
    const LLType* voidPtrTy = rt_ptr(byteTy);
    const LLType* voidArrayTy = rt_array(byteTy);
    const LLType* voidArrayPtrTy = getPtrToType(voidArrayTy);
    const LLType* stringTy = voidArrayTy;
    const LLType* wstringTy = rt_array(shortTy);
    const LLType* dstringTy = rt_array(intTy);

    Logger::println("building class types");
    const LLType* objectTy = DtoType(ClassDeclaration::object->type);
    const LLType* classInfoTy = DtoType(ClassDeclaration::classinfo->type);
    const LLType* typeInfoTy = DtoType(Type::typeinfo->type);

    Logger::println("building aa type");
    const LLType* aaTy = rt_ptr(llvm::OpaqueType::get(gIR->context()));

    Logger::println("building functions");

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Construct some attribute lists used below (possibly multiple times)
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
        Attr_ReadOnly_1_4_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(4, NoCapture),
        Attr_ReadOnly_NoUnwind_1_NoCapture
            = Attr_ReadOnly_1_NoCapture.addAttr(~0U, NoUnwind),
        Attr_ReadNone
            = NoAttrs.addAttr(~0U, ReadNone),
        Attr_1_NoCapture
            = NoAttrs.addAttr(1, NoCapture),
        Attr_NoAlias_1_NoCapture
            = Attr_1_NoCapture.addAttr(0, NoAlias),
        Attr_NoAlias_3_NoCapture
            = Attr_NoAlias.addAttr(3, NoCapture),
        Attr_1_2_NoCapture
            = Attr_1_NoCapture.addAttr(2, NoCapture),
        Attr_1_3_NoCapture
            = Attr_1_NoCapture.addAttr(3, NoCapture),
        Attr_1_4_NoCapture
            = Attr_1_NoCapture.addAttr(4, NoCapture);

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_assert( char[] file, uint line )
    {
        llvm::StringRef fname("_d_assert");
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // D1:
    // void _d_array_bounds( char[] file, uint line )
    // void _d_switch_error( char[] file, uint line )
    // D2:
    // void _d_array_bounds(ModuleInfo* m, uint line)
    // void _d_switch_error(ModuleInfo* m, uint line)
    {
        llvm::StringRef fname("_d_array_bounds");
        llvm::StringRef fname2("_d_switch_error");
        std::vector<const LLType*> types;
#if DMDV2
        types.push_back(getPtrToType(DtoType(Module::moduleinfo->type)));
#else
        types.push_back(stringTy);
#endif
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_assert_msg( char[] msg, char[] file, uint line )
    {
        llvm::StringRef fname("_d_assert_msg");
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////


    // void* _d_allocmemory(size_t sz)
    {
        llvm::StringRef fname("_d_allocmemory");
        std::vector<const LLType*> types;
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
                ->setAttributes(Attr_NoAlias);
    }

    // void* _d_allocmemoryT(TypeInfo ti)
    {
        llvm::StringRef fname("_d_allocmemoryT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
    }
#if DMDV1
    // void* _d_newarrayT(TypeInfo ti, size_t length)
    // void* _d_newarrayiT(TypeInfo ti, size_t length)
    // void* _d_newarrayvT(TypeInfo ti, size_t length)
    {
        llvm::StringRef fname("_d_newarrayT");
        llvm::StringRef fname2("_d_newarrayiT");
        llvm::StringRef fname3("_d_newarrayvT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_NoAlias);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M)
            ->setAttributes(Attr_NoAlias);
    }
    // void* _d_newarraymT(TypeInfo ti, size_t length, size_t* dims)
    // void* _d_newarraymiT(TypeInfo ti, size_t length, size_t* dims)
    // void* _d_newarraymvT(TypeInfo ti, size_t length, size_t* dims)
    {
        llvm::StringRef fname("_d_newarraymT");
        llvm::StringRef fname2("_d_newarraymiT");
        llvm::StringRef fname3("_d_newarraymvT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(rt_ptr(sizeTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias_3_NoCapture);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_NoAlias_3_NoCapture);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M)
            ->setAttributes(Attr_NoAlias_3_NoCapture);
    }
#else
    // void[] _d_newarrayT(TypeInfo ti, size_t length)
    // void[] _d_newarrayiT(TypeInfo ti, size_t length)
    {
        llvm::StringRef fname("_d_newarrayT");
        llvm::StringRef fname2("_d_newarrayiT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }
    // void[] _d_newarraymT(TypeInfo ti, size_t length, size_t* dims)
    // void[] _d_newarraymiT(TypeInfo ti, size_t length, size_t* dims)
    {
        llvm::StringRef fname("_d_newarraymT");
        llvm::StringRef fname2("_d_newarraymiT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, true);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }
#endif

    // D1:
    // void* _d_arraysetlengthT(TypeInfo ti, size_t newlength, size_t plength, void* pdata)
    // void* _d_arraysetlengthiT(TypeInfo ti, size_t newlength, size_t plength, void* pdata)
    // D2:
    // void[] _d_arraysetlengthT(TypeInfo ti, size_t newlength, void[] *array)
    // void[] _d_arraysetlengthiT(TypeInfo ti, size_t newlength, void[] *array)
    {
        llvm::StringRef fname("_d_arraysetlengthT");
        llvm::StringRef fname2("_d_arraysetlengthiT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
#if DMDV2
        types.push_back(voidArrayPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
#else
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
#endif
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

#if DMDV2
    // byte[] _d_arrayappendcTX(TypeInfo ti, ref byte[] px, size_t n)
    {
        llvm::StringRef fname("_d_arrayappendcTX");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidArrayPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendT(TypeInfo ti, byte[]* px, byte[] y)
    {
        llvm::StringRef fname("_d_arrayappendT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidArrayPtrTy);
        types.push_back(voidArrayTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendcd(ref char[] x, dchar c)
    {
        llvm::StringRef fname("_d_arrayappendcd");
        std::vector<const LLType*> types;
        types.push_back(getPtrToType(stringTy));
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // void[] _d_arrayappendwd(ref wchar[] x, dchar c)
    {
        llvm::StringRef fname("_d_arrayappendwd");
        std::vector<const LLType*> types;
        types.push_back(getPtrToType(wstringTy));
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
    // byte[] _d_arraycatT(TypeInfo ti, byte[] x, byte[] y)
    {
        llvm::StringRef fname("_d_arraycatT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidArrayTy);
        types.push_back(voidArrayTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
#else // DMDV1
    // byte[] _d_arrayappendcT(TypeInfo ti, void* array, void* element)
    {
        llvm::StringRef fname("_d_arrayappendcT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
#endif

    // Object _d_allocclass(ClassInfo ci)
    {
        llvm::StringRef fname(_d_allocclass);
        std::vector<const LLType*> types;
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
    }

#if DMDV2

    // void _d_delarray_t(Array *p, TypeInfo ti)
    {
        llvm::StringRef fname("_d_delarray_t");
        std::vector<const LLType*> types;
        types.push_back(voidArrayPtrTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

#else

    // void _d_delarray(size_t plength, void* pdata)
    {
        llvm::StringRef fname("_d_delarray");
        std::vector<const LLType*> types;
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

#endif

    // D1:
    // void _d_delmemory(void* p)
    // void _d_delinterface(void* p)
    // void _d_callfinalizer(void* p)
    // D2:
    // void _d_delmemory(void **p)
    // void _d_delinterface(void **p)
    // void _d_callfinalizer(void *p)
    {
        llvm::StringRef fname("_d_delmemory");
        llvm::StringRef fname2("_d_delinterface");
        llvm::StringRef fname3("_d_callfinalizer");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }

    // D1: void _d_delclass(Object p)
    // D2: void _d_delclass(Object* p)
    {
        llvm::StringRef fname("_d_delclass");
        std::vector<const LLType*> types;
#if DMDV2
        types.push_back(rt_ptr(objectTy));
#else
        types.push_back(objectTy);
#endif
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    #define ARRAY_INIT(TY,suffix) \
    { \
        std::string fname = (llvm::StringRef("_d_array_init_") + llvm::StringRef(suffix)).str(); \
        std::vector<const LLType*> types; \
        types.push_back(rt_ptr(TY)); \
        types.push_back(sizeTy); \
        types.push_back(TY); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M) \
            ->setAttributes(Attr_1_NoCapture); \
    }

    ARRAY_INIT(shortTy,"i16")
    ARRAY_INIT(intTy,"i32")
    ARRAY_INIT(longTy,"i64")
    ARRAY_INIT(floatTy,"float")
    ARRAY_INIT(doubleTy,"double")
    ARRAY_INIT(realTy,"real")
    ARRAY_INIT(cfloatTy,"cfloat")
    ARRAY_INIT(cdoubleTy,"cdouble")
    ARRAY_INIT(crealTy,"creal")
    ARRAY_INIT(voidPtrTy,"pointer")

    #undef ARRAY_INIT

    // array init mem
    // void _d_array_init_mem(void* a, size_t na, void* v, size_t nv)
    // +
    // array slice copy when assertions are on!
    // void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t srclen)
    {
        llvm::StringRef fname("_d_array_init_mem");
        llvm::StringRef fname2("_d_array_slice_copy");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_3_NoCapture);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_1_3_NoCapture);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _aApplycd1(char[] aa, dg_t dg)
    #define STR_APPLY1(TY,a,b) \
    { \
        llvm::StringRef fname(a); \
        llvm::StringRef fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY1(stringTy, "_aApplycw1", "_aApplycd1")
    STR_APPLY1(wstringTy, "_aApplywc1", "_aApplywd1")
    STR_APPLY1(dstringTy, "_aApplydc1", "_aApplydw1")
    #undef STR_APPLY

    // int _aApplycd2(char[] aa, dg2_t dg)
    #define STR_APPLY2(TY,a,b) \
    { \
        llvm::StringRef fname(a); \
        llvm::StringRef fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY2(stringTy, "_aApplycw2", "_aApplycd2")
    STR_APPLY2(wstringTy, "_aApplywc2", "_aApplywd2")
    STR_APPLY2(dstringTy, "_aApplydc2", "_aApplydw2")
    #undef STR_APPLY2

    #define STR_APPLY_R1(TY,a,b) \
    { \
        llvm::StringRef fname(a); \
        llvm::StringRef fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY_R1(stringTy, "_aApplyRcw1", "_aApplyRcd1")
    STR_APPLY_R1(wstringTy, "_aApplyRwc1", "_aApplyRwd1")
    STR_APPLY_R1(dstringTy, "_aApplyRdc1", "_aApplyRdw1")
    #undef STR_APPLY

    #define STR_APPLY_R2(TY,a,b) \
    { \
        llvm::StringRef fname(a); \
        llvm::StringRef fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
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
    {
        llvm::StringRef fname("_d_array_cast_len");
        std::vector<const LLType*> types;
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadNone);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

#if DMDV2

    // void[] _d_arrayassign(TypeInfo ti, void[] from, void[] to)
    // void[] _d_arrayctor(TypeInfo ti, void[] from, void[] to)
    {
        llvm::StringRef fname("_d_arrayassign");
        llvm::StringRef fname2("_d_arrayctor");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidArrayTy);
        types.push_back(voidArrayTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidArrayTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void* _d_arraysetassign(void* p, void* value, size_t count, TypeInfo ti)
    // void* _d_arraysetctor(void* p, void* value, size_t count, TypeInfo ti)
    {
        llvm::StringRef fname("_d_arraysetassign");
        llvm::StringRef fname2("_d_arraysetctor");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_NoAlias);
    }

#endif

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // cast to object
    // Object _d_toObject(void* p)
    {
        llvm::StringRef fname("_d_toObject");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // cast interface
    // Object _d_interface_cast(void* p, ClassInfo c)
    {
        llvm::StringRef fname("_d_interface_cast");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // dynamic cast
    // Object _d_dynamic_cast(Object o, ClassInfo c)
    {
        llvm::StringRef fname("_d_dynamic_cast");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
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
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(stringTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // wchar[] _adReverseWchar(wchar[] a)
    // wchar[] _adSortWchar(wchar[] a)
    {
        llvm::StringRef fname("_adReverseWchar");
        llvm::StringRef fname2("_adSortWchar");
        std::vector<const LLType*> types;
        types.push_back(wstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(wstringTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void[] _adReverse(void[] a, size_t szelem)
    {
        llvm::StringRef fname("_adReverse");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(rt_array(byteTy), types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoUnwind);
    }

    // void[] _adDupT(TypeInfo ti, void[] a)
    {
        llvm::StringRef fname("_adDupT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(rt_array(byteTy), types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _adEq(void[] a1, void[] a2, TypeInfo ti)
    // int _adCmp(void[] a1, void[] a2, TypeInfo ti)
    {
        llvm::StringRef fname(_adEq);
        llvm::StringRef fname2(_adCmp);
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _adCmpChar(void[] a1, void[] a2)
    {
        llvm::StringRef fname("_adCmpChar");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind);
    }

    // void[] _adSort(void[] a, TypeInfo ti)
    {
        llvm::StringRef fname("_adSort");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(rt_array(byteTy), types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // size_t _aaLen(AA aa)
    {
        llvm::StringRef fname("_aaLen");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_NoUnwind_1_NoCapture);
    }

    // D1:
    // void* _aaGet(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    // D2:
    // void* _aaGetX(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    {
#if DMDV2
        llvm::StringRef fname("_aaGetX");
#else
        llvm::StringRef fname("_aaGet");
#endif
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_4_NoCapture);
    }

    // D1:
    // void* _aaIn(AA aa, TypeInfo keyti, void* pkey)
    // D2:
    // void* _aaInX(AA aa, TypeInfo keyti, void* pkey)
    {
#if DMDV2
        llvm::StringRef fname("_aaInX");
#else
        llvm::StringRef fname("_aaIn");
#endif
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly_1_3_NoCapture);
    }

    // D1:
    // void _aaDel(AA aa, TypeInfo keyti, void* pkey)
    // D2:
    // void _aaDelX(AA aa, TypeInfo keyti, void* pkey)
    {
#if DMDV2
        llvm::StringRef fname("_aaDelX");
#else
        llvm::StringRef fname("_aaDel");
#endif
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_3_NoCapture);
    }

    // void[] _aaValues(AA aa, size_t keysize, size_t valuesize)
    {
        llvm::StringRef fname("_aaValues");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(rt_array(byteTy), types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias_1_NoCapture);
    }

    // void* _aaRehash(AA* paa, TypeInfo keyti)
    {
        llvm::StringRef fname("_aaRehash");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void[] _aaKeys(AA aa, size_t keysize)
    {
        llvm::StringRef fname("_aaKeys");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(rt_array(byteTy), types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_NoAlias_1_NoCapture);
    }

    // int _aaApply(AA aa, size_t keysize, dg_t dg)
    {
        llvm::StringRef fname("_aaApply");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg1());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_NoCapture);
    }

    // int _aaApply2(AA aa, size_t keysize, dg2_t dg)
    {
        llvm::StringRef fname("_aaApply2");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg2());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_NoCapture);
    }

#if DMDV2
    // int _aaEqual(TypeInfo_AssociativeArray ti, AA e1, AA e2)
    {
        llvm::StringRef fname("_aaEqual");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(aaTy);
        types.push_back(aaTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_2_NoCapture);
    }
    // BB* _d_assocarrayliteralTX(TypeInfo_AssociativeArray ti, void[] keys, void[] values)
    {
        llvm::StringRef fname("_d_assocarrayliteralTX");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(voidArrayTy);
        types.push_back(voidArrayTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
#else
    // int _aaEq(AA aa, AA ab, TypeInfo_AssociativeArray ti)
    {
        llvm::StringRef fname("_aaEq");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_2_NoCapture);
    }
#endif

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _moduleCtor()
    // void _moduleDtor()
    {
        llvm::StringRef fname("_moduleCtor");
        llvm::StringRef fname2("_moduleDtor");
        std::vector<const LLType*> types;
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_throw_exception(Object e)
    {
        llvm::StringRef fname("_d_throw_exception");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_switch_string(char[][] table, char[] ca)
    {
        llvm::StringRef fname("_d_switch_string");
        std::vector<const LLType*> types;
        types.push_back(rt_array(stringTy));
        types.push_back(stringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _d_switch_ustring(wchar[][] table, wchar[] ca)
    {
        llvm::StringRef fname("_d_switch_ustring");
        std::vector<const LLType*> types;
        types.push_back(rt_array(wstringTy));
        types.push_back(wstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

    // int _d_switch_dstring(dchar[][] table, dchar[] ca)
    {
        llvm::StringRef fname("_d_switch_dstring");
        std::vector<const LLType*> types;
        types.push_back(rt_array(dstringTy));
        types.push_back(dstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_ReadOnly);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_criticalenter(D_CRITICAL_SECTION *dcs)
    // void _d_criticalexit(D_CRITICAL_SECTION *dcs)
    {
        llvm::StringRef fname("_d_criticalenter");
        llvm::StringRef fname2("_d_criticalexit");
        std::vector<const LLType*> types;
        types.push_back(rt_ptr(DtoMutexType()));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_monitorenter(Object h)
    // void _d_monitorexit(Object h)
    {
        llvm::StringRef fname("_d_monitorenter");
        llvm::StringRef fname2("_d_monitorexit");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)
            ->setAttributes(Attr_1_NoCapture);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)
            ->setAttributes(Attr_1_NoCapture);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_eh_personality(int ver, int actions, ulong eh_class, ptr eh_info, ptr context)
    {
        llvm::StringRef fname("_d_eh_personality");
        std::vector<const LLType*> types;
        types.push_back(intTy);
        types.push_back(intTy);
        types.push_back(longTy);
        types.push_back(voidPtrTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_eh_resume_unwind(ptr exc_struct)
    {
        llvm::StringRef fname("_d_eh_resume_unwind");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_invariant(Object o)
    {
        llvm::StringRef fname("_d_invariant");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

#if DMDV2
    // void _d_hidden_func()
    {
        llvm::StringRef fname("_d_hidden_func");
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
#endif
}
