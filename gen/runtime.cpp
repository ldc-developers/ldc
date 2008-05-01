#include <cassert>

#include "gen/llvm.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"

#include "root.h"
#include "mars.h"
#include "lexer.h"
#include "dsymbol.h"
#include "mtype.h"
#include "aggregate.h"

#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/irstate.h"

static llvm::Module* M = NULL;
static bool runtime_failed = false;

static void LLVM_D_BuildRuntimeModule();

//////////////////////////////////////////////////////////////////////////////////////////////////

bool LLVM_D_InitRuntime()
{
    Logger::println("*** Initializing D runtime declarations ***");
    LOG_SCOPE;

    LLVM_D_BuildRuntimeModule();
    return true;

    /*
    if (!global.params.runtimeImppath) {
        error("You must set the runtime import path with -E");
        fatal();
    }
    std::string filename(global.params.runtimeImppath);
    filename.append("/llvmdcore.bc");
    llvm::MemoryBuffer* buffer = llvm::MemoryBuffer::getFile(filename.c_str(), filename.length());
    if (!buffer) {
        Logger::println("Failed to load runtime library from disk");
        runtime_failed = true;
        return false;
    }

    std::string errstr;
    bool retval = false;
    M = llvm::ParseBitcodeFile(buffer, &errstr);
    if (M) {
        retval = true;
    }
    else {
        Logger::println("Failed to load runtime: %s", errstr.c_str());
        runtime_failed = true;
    }

    delete buffer;
    return retval;
    */
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
    if (global.params.noruntime) {
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
    return llvm::cast<llvm::Function>(target->getOrInsertFunction(name, fnty));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name)
{
    llvm::GlobalVariable* gv = target->getNamedGlobal(name);
    if (gv) {
        return gv;
    }

    if (global.params.noruntime) {
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
    return new llvm::GlobalVariable(t->getElementType(),g->isConstant(),g->getLinkage(),NULL,g->getName(),target);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static const llvm::Type* rt_ptr(const llvm::Type* t)
{
    return getPtrToType(t);
}

static const llvm::Type* rt_array(const llvm::Type* elemty)
{
    std::vector<const llvm::Type*> t;
    t.push_back(DtoSize_t());
    t.push_back(rt_ptr(elemty));
    return rt_ptr(llvm::StructType::get(t));
}

static const llvm::Type* rt_dg1()
{
    std::vector<const llvm::Type*> types;
    types.push_back(rt_ptr(llvm::Type::Int8Ty));
    types.push_back(rt_ptr(llvm::Type::Int8Ty));
    const llvm::FunctionType* fty = llvm::FunctionType::get(llvm::Type::Int32Ty, types, false);

    std::vector<const llvm::Type*> t;
    t.push_back(rt_ptr(llvm::Type::Int8Ty));
    t.push_back(rt_ptr(fty));
    return rt_ptr(llvm::StructType::get(t));
}

static const llvm::Type* rt_dg2()
{
    std::vector<const llvm::Type*> types;
    types.push_back(rt_ptr(llvm::Type::Int8Ty));
    types.push_back(rt_ptr(llvm::Type::Int8Ty));
    types.push_back(rt_ptr(llvm::Type::Int8Ty));
    const llvm::FunctionType* fty = llvm::FunctionType::get(llvm::Type::Int32Ty, types, false);

    std::vector<const llvm::Type*> t;
    t.push_back(rt_ptr(llvm::Type::Int8Ty));
    t.push_back(rt_ptr(fty));
    return rt_ptr(llvm::StructType::get(t));
}

static void LLVM_D_BuildRuntimeModule()
{
    M = new llvm::Module("llvmdc internal runtime");

    const llvm::Type* voidTy = llvm::Type::VoidTy;
    const llvm::Type* boolTy = llvm::Type::Int1Ty;
    const llvm::Type* byteTy = llvm::Type::Int8Ty;
    const llvm::Type* shortTy = llvm::Type::Int16Ty;
    const llvm::Type* intTy = llvm::Type::Int32Ty;
    const llvm::Type* longTy = llvm::Type::Int64Ty;
    const llvm::Type* floatTy = llvm::Type::FloatTy;
    const llvm::Type* doubleTy = llvm::Type::DoubleTy;
    const llvm::Type* sizeTy = DtoSize_t();
    const llvm::Type* voidPtrTy = rt_ptr(byteTy);
    const llvm::Type* stringTy = rt_array(byteTy);
    const llvm::Type* wstringTy = rt_array(shortTy);
    const llvm::Type* dstringTy = rt_array(intTy);
    const llvm::Type* objectTy = rt_ptr(gIR->irType[ClassDeclaration::object->type].type->get());
    const llvm::Type* classInfoTy = rt_ptr(gIR->irType[ClassDeclaration::classinfo->type].type->get());
    const llvm::Type* typeInfoTy = rt_ptr(gIR->irType[Type::typeinfo->type].type->get());
    const llvm::Type* aaTy = rt_ptr(llvm::OpaqueType::get());

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_assert( char[] file, uint line )
    // void _d_array_bounds( char[] file, uint line )
    // void _d_switch_error( char[] file, uint line )
    {
        std::string fname("_d_assert");
        std::string fname2("_d_array_bounds");
        std::string fname3("_d_switch_error");
        std::vector<const llvm::Type*> types;
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }

    // void _d_assert_msg( char[] msg, char[] file, uint line )
    {
        std::string fname("_d_assert_msg");
        std::vector<const llvm::Type*> types;
        types.push_back(stringTy);
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // realloc
    // void* _d_realloc(void* ptr, size_t n)
    {
        std::string fname("_d_realloc");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // free
    // void _d_free(void* ptr)
    {
        std::string fname("_d_free");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_newclass(ClassInfo ci)
    {
        std::string fname("_d_newclass");
        std::vector<const llvm::Type*> types;
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    #define ARRAY_INIT(TY,suffix) \
    { \
        std::string fname("_d_array_init_"); \
        fname.append(suffix); \
        std::vector<const llvm::Type*> types; \
        types.push_back(rt_ptr(TY)); \
        types.push_back(sizeTy); \
        types.push_back(TY); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
    }

    ARRAY_INIT(boolTy,"i1")
    ARRAY_INIT(byteTy,"i8")
    ARRAY_INIT(shortTy,"i16")
    ARRAY_INIT(intTy,"i32")
    ARRAY_INIT(longTy,"i64")
    ARRAY_INIT(floatTy,"float")
    ARRAY_INIT(doubleTy,"double")
    ARRAY_INIT(voidPtrTy,"pointer")

    #undef ARRAY_INIT

    // array init mem
    // void _d_array_init_mem(void* a, size_t na, void* v, size_t nv)
    {
        std::string fname("_d_array_init_mem");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    #define STR_APPLY1(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const llvm::Type*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY1(stringTy, "_aApplycw1", "_aApplycd1")
    STR_APPLY1(wstringTy, "_aApplywc1", "_aApplywd1")
    STR_APPLY1(dstringTy, "_aApplydc1", "_aApplydw1")
    #undef STR_APPLY

    #define STR_APPLY2(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const llvm::Type*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY2(stringTy, "_aApplycw2", "_aApplycd2")
    STR_APPLY2(wstringTy, "_aApplywc2", "_aApplywd2")
    STR_APPLY2(dstringTy, "_aApplydc2", "_aApplydw2")
    #undef STR_APPLY2

    #define STR_APPLY_R1(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const llvm::Type*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
    }
    STR_APPLY_R1(stringTy, "_aApplyRcw1", "_aApplyRcd1")
    STR_APPLY_R1(wstringTy, "_aApplyRwc1", "_aApplyRwd1")
    STR_APPLY_R1(dstringTy, "_aApplyRdc1", "_aApplyRdw1")
    #undef STR_APPLY

    #define STR_APPLY_R2(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const llvm::Type*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M); \
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
        std::string fname("_d_array_cast_len");
        std::vector<const llvm::Type*> types;
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // builds the d string[] for the D main args from the C main args
    // void _d_main_args(uint n, char** args, ref char[][] res)
    {
        std::string fname("_d_main_args");
        std::vector<const llvm::Type*> types;
        types.push_back(intTy);
        types.push_back(rt_ptr(rt_ptr(byteTy)));
        types.push_back(rt_array(stringTy->getContainedType(0)));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // cast to object
    // Object _d_toObject(void* p)
    {
        std::string fname("_d_toObject");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // cast interface
    // Object _d_interface_cast(void* p, ClassInfo c)
    {
        std::string fname("_d_interface_cast");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // dynamic cast
    // Object _d_dynamic_cast(Object o, ClassInfo c)
    {
        std::string fname("_d_dynamic_cast");
        std::vector<const llvm::Type*> types;
        types.push_back(objectTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // char[] _adReverseChar(char[] a)
    // char[] _adSortChar(char[] a)
    {
        std::string fname("_adReverseChar");
        std::string fname2("_adSortChar");
        std::vector<const llvm::Type*> types;
        types.push_back(stringTy);
        types.push_back(stringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // wchar[] _adReverseWchar(wchar[] a)
    // wchar[] _adSortWchar(wchar[] a)
    {
        std::string fname("_adReverseWchar");
        std::string fname2("_adSortWchar");
        std::vector<const llvm::Type*> types;
        types.push_back(wstringTy);
        types.push_back(wstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // Array _adReverse(Array a, size_t szelem)
    {
        std::string fname("_adReverse");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Array _adDupT(TypeInfo ti, Array a)
    {
        std::string fname("_adDupT");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _adEq(Array a1, Array a2, TypeInfo ti)
    // int _adCmp(Array a1, Array a2, TypeInfo ti)
    {
        std::string fname("_adEq");
        std::string fname2("_adCmp");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // int _adCmpChar(Array a1, Array a2)
    {
        std::string fname("_adCmpChar");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Array _adSort(Array a, TypeInfo ti)
    {
        std::string fname("_adSort");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // size_t _aaLen(AA aa)
    {
        std::string fname("_aaLen");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaGet(AA* aa, TypeInfo keyti, void* pkey, size_t valuesize)
    {
        std::string fname("_aaGet");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaGetRvalue(AA aa, TypeInfo keyti, size_t valuesize, void* pkey)
    {
        std::string fname("_aaGetRvalue");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaIn(AA aa, TypeInfo keyti, void* pkey)
    {
        std::string fname("_aaIn");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _aaDel(AA aa, TypeInfo keyti, void* pkey)
    {
        std::string fname("_aaDel");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // ArrayRet_t _aaValues(AA aa, size_t keysize, size_t valuesize)
    {
        std::string fname("_aaValues");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaRehash(AA* paa, TypeInfo keyti)
    {
        std::string fname("_aaRehash");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // ArrayRet_t _aaKeys(AA aa, size_t keysize)
    {
        std::string fname("_aaKeys");
        std::vector<const llvm::Type*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(aaTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _aaApply(AA aa, size_t keysize, dg_t dg)
    {
        std::string fname("_aaApply");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg1());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _aaApply2(AA aa, size_t keysize, dg2_t dg)
    {
        std::string fname("_aaApply2");
        std::vector<const llvm::Type*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg1());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _moduleCtor()
    // void _moduleDtor()
    {
        std::string fname("_moduleCtor");
        std::string fname2("_moduleDtor");
        std::vector<const llvm::Type*> types;
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Object _d_toObject(void* p)
    {
        std::string fname("_d_toObject");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_dynamic_cast(Object o, ClassInfo c)
    {
        std::string fname("_d_dynamic_cast");
        std::vector<const llvm::Type*> types;
        types.push_back(objectTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_interface_cast(void* p, ClassInfo c)
    {
        std::string fname("_d_interface_cast");
        std::vector<const llvm::Type*> types;
        types.push_back(voidPtrTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_throw_exception(Object e)
    {
        std::string fname("_d_throw_exception");
        std::vector<const llvm::Type*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        new llvm::Function(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
}






