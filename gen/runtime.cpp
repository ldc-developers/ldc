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

static const LLType* rt_ptr(const LLType* t)
{
    return getPtrToType(t);
}

static const LLType* rt_array(const LLType* elemty)
{
    std::vector<const LLType*> t;
    t.push_back(DtoSize_t());
    t.push_back(rt_ptr(elemty));
    return rt_ptr(llvm::StructType::get(t));
}

static const LLType* rt_array2(const LLType* elemty)
{
    std::vector<const LLType*> t;
    t.push_back(DtoSize_t());
    t.push_back(rt_ptr(elemty));
    return llvm::StructType::get(t);
}

static const LLType* rt_dg1()
{
    std::vector<const LLType*> types;
    types.push_back(rt_ptr(LLType::Int8Ty));
    types.push_back(rt_ptr(LLType::Int8Ty));
    const llvm::FunctionType* fty = llvm::FunctionType::get(LLType::Int32Ty, types, false);

    std::vector<const LLType*> t;
    t.push_back(rt_ptr(LLType::Int8Ty));
    t.push_back(rt_ptr(fty));
    return rt_ptr(llvm::StructType::get(t));
}

static const LLType* rt_dg2()
{
    std::vector<const LLType*> types;
    types.push_back(rt_ptr(LLType::Int8Ty));
    types.push_back(rt_ptr(LLType::Int8Ty));
    types.push_back(rt_ptr(LLType::Int8Ty));
    const llvm::FunctionType* fty = llvm::FunctionType::get(LLType::Int32Ty, types, false);

    std::vector<const LLType*> t;
    t.push_back(rt_ptr(LLType::Int8Ty));
    t.push_back(rt_ptr(fty));
    return rt_ptr(llvm::StructType::get(t));
}

static void LLVM_D_BuildRuntimeModule()
{
    M = new llvm::Module("llvmdc internal runtime");

    const LLType* voidTy = LLType::VoidTy;
    const LLType* boolTy = LLType::Int1Ty;
    const LLType* byteTy = LLType::Int8Ty;
    const LLType* shortTy = LLType::Int16Ty;
    const LLType* intTy = LLType::Int32Ty;
    const LLType* longTy = LLType::Int64Ty;
    const LLType* sizeTy = DtoSize_t();

    const LLType* floatTy = LLType::FloatTy;
    const LLType* doubleTy = LLType::DoubleTy;
    const LLType* realTy;
    if (global.params.cpu == ARCHx86)
        realTy = LLType::X86_FP80Ty;
    else
        realTy = LLType::DoubleTy;

    const LLType* cfloatTy = llvm::StructType::get(floatTy, floatTy, 0);
    const LLType* cdoubleTy = llvm::StructType::get(doubleTy, doubleTy, 0);
    const LLType* crealTy = llvm::StructType::get(realTy, realTy, 0);

    const LLType* voidPtrTy = rt_ptr(byteTy);
    const LLType* stringTy = rt_array(byteTy);
    const LLType* wstringTy = rt_array(shortTy);
    const LLType* dstringTy = rt_array(intTy);
    const LLType* objectTy = rt_ptr(ClassDeclaration::object->type->ir.type->get());
    const LLType* classInfoTy = rt_ptr(ClassDeclaration::classinfo->type->ir.type->get());
    const LLType* typeInfoTy = rt_ptr(Type::typeinfo->type->ir.type->get());
    const LLType* aaTy = rt_ptr(llvm::OpaqueType::get());

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
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M)->setAttributes(palist);
    }

    // void _d_assert_msg( char[] msg, char[] file, uint line )
    {
        std::string fname("_d_assert_msg");
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        types.push_back(stringTy);
        types.push_back(intTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void* _d_allocmemoryT(TypeInfo ti)
    {
        std::string fname("_d_allocmemoryT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _d_newarrayT(TypeInfo ti, size_t length)
    // void* _d_newarrayiT(TypeInfo ti, size_t length)
    // void* _d_newarrayvT(TypeInfo ti, size_t length)
    {
        std::string fname("_d_newarrayT");
        std::string fname2("_d_newarrayiT");
        std::string fname3("_d_newarrayvT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }

    // void* _d_newarraymT(TypeInfo ti, size_t length, size_t* dims)
    // void* _d_newarraymiT(TypeInfo ti, size_t length, size_t* dims)
    // void* _d_newarraymvT(TypeInfo ti, size_t length, size_t* dims)
    {
        std::string fname("_d_newarraymT");
        std::string fname2("_d_newarraymiT");
        std::string fname3("_d_newarraymvT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(rt_ptr(sizeTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }

    // void* _d_arraysetlengthT(TypeInfo ti, size_t newlength, size_t plength, void* pdata)
    // void* _d_arraysetlengthiT(TypeInfo ti, size_t newlength, size_t plength, void* pdata)
    {
        std::string fname("_d_arraysetlengthT");
        std::string fname2("_d_arraysetlengthiT");
        std::vector<const LLType*> types;
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // Object _d_newclass(ClassInfo ci)
    {
        std::string fname("_d_newclass");
        std::vector<const LLType*> types;
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_delarray(size_t plength, void* pdata)
    {
        std::string fname("_d_delarray");
        std::vector<const LLType*> types;
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _d_delmemory(void* p)
    // void _d_delinterface(void* p)
    // void _d_callfinalizer(void* p)
    {
        std::string fname("_d_delmemory");
        std::string fname2("_d_delinterface");
        std::string fname3("_d_callfinalizer");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname3, M);
    }

    // void _d_delclass(Object p)
    {
        std::string fname("_d_delclass");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    #define ARRAY_INIT(TY,suffix) \
    { \
        std::string fname("_d_array_init_"); \
        fname.append(suffix); \
        std::vector<const LLType*> types; \
        types.push_back(rt_ptr(TY)); \
        types.push_back(sizeTy); \
        types.push_back(TY); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M); \
    }

    ARRAY_INIT(boolTy,"i1")
    ARRAY_INIT(byteTy,"i8")
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
    {
        std::string fname("_d_array_init_mem");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _aApplycd1(char[] aa, dg_t dg)
    #define STR_APPLY1(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::AttrListPtr palist; \
        palist = palist.addAttr(1, llvm::Attribute::ByVal); \
        palist = palist.addAttr(2, llvm::Attribute::ByVal); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist); \
    }
    STR_APPLY1(stringTy, "_aApplycw1", "_aApplycd1")
    STR_APPLY1(wstringTy, "_aApplywc1", "_aApplywd1")
    STR_APPLY1(dstringTy, "_aApplydc1", "_aApplydw1")
    #undef STR_APPLY

    // int _aApplycd2(char[] aa, dg2_t dg)
    #define STR_APPLY2(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::AttrListPtr palist; \
        palist = palist.addAttr(1, llvm::Attribute::ByVal); \
        palist = palist.addAttr(2, llvm::Attribute::ByVal); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist); \
    }
    STR_APPLY2(stringTy, "_aApplycw2", "_aApplycd2")
    STR_APPLY2(wstringTy, "_aApplywc2", "_aApplywd2")
    STR_APPLY2(dstringTy, "_aApplydc2", "_aApplydw2")
    #undef STR_APPLY2

    #define STR_APPLY_R1(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg1()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::AttrListPtr palist; \
        palist = palist.addAttr(1, llvm::Attribute::ByVal); \
        palist = palist.addAttr(2, llvm::Attribute::ByVal); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist); \
    }
    STR_APPLY_R1(stringTy, "_aApplyRcw1", "_aApplyRcd1")
    STR_APPLY_R1(wstringTy, "_aApplyRwc1", "_aApplyRwd1")
    STR_APPLY_R1(dstringTy, "_aApplyRdc1", "_aApplyRdw1")
    #undef STR_APPLY

    #define STR_APPLY_R2(TY,a,b) \
    { \
        std::string fname(a); \
        std::string fname2(b); \
        std::vector<const LLType*> types; \
        types.push_back(TY); \
        types.push_back(rt_dg2()); \
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false); \
        llvm::AttrListPtr palist; \
        palist = palist.addAttr(1, llvm::Attribute::ByVal); \
        palist = palist.addAttr(2, llvm::Attribute::ByVal); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist); \
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist); \
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
        std::vector<const LLType*> types;
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // builds the d string[] for the D main args from the C main args
    // void _d_main_args(uint n, char** args, ref char[][] res)
    {
        std::string fname("_d_main_args");
        std::vector<const LLType*> types;
        types.push_back(intTy);
        types.push_back(rt_ptr(rt_ptr(byteTy)));
        types.push_back(rt_array(stringTy->getContainedType(0)));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // cast to object
    // Object _d_toObject(void* p)
    {
        std::string fname("_d_toObject");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // cast interface
    // Object _d_interface_cast(void* p, ClassInfo c)
    {
        std::string fname("_d_interface_cast");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // dynamic cast
    // Object _d_dynamic_cast(Object o, ClassInfo c)
    {
        std::string fname("_d_dynamic_cast");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // char[] _adReverseChar(char[] a)
    // char[] _adSortChar(char[] a)
    {
        std::string fname("_adReverseChar");
        std::string fname2("_adSortChar");
        std::vector<const LLType*> types;
        types.push_back(stringTy);
        types.push_back(stringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist);
    }

    // wchar[] _adReverseWchar(wchar[] a)
    // wchar[] _adSortWchar(wchar[] a)
    {
        std::string fname("_adReverseWchar");
        std::string fname2("_adSortWchar");
        std::vector<const LLType*> types;
        types.push_back(wstringTy);
        types.push_back(wstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist);
    }

    // Array _adReverse(Array a, size_t szelem)
    {
        std::string fname("_adReverse");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // Array _adDupT(TypeInfo ti, Array a)
    {
        std::string fname("_adDupT");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(3, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // int _adEq(Array a1, Array a2, TypeInfo ti)
    // int _adCmp(Array a1, Array a2, TypeInfo ti)
    {
        std::string fname("_adEq");
        std::string fname2("_adCmp");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M)->setAttributes(palist);
    }

    // int _adCmpChar(Array a1, Array a2)
    {
        std::string fname("_adCmpChar");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // Array _adSort(Array a, TypeInfo ti)
    {
        std::string fname("_adSort");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(rt_array(byteTy));
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // size_t _aaLen(AA aa)
    {
        std::string fname("_aaLen");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(sizeTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaGet(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    {
        std::string fname("_aaGet");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaGetRvalue(AA aa, TypeInfo keyti, size_t valuesize, void* pkey)
    {
        std::string fname("_aaGetRvalue");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(sizeTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaIn(AA aa, TypeInfo keyti, void* pkey)
    {
        std::string fname("_aaIn");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void _aaDel(AA aa, TypeInfo keyti, void* pkey)
    {
        std::string fname("_aaDel");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // ArrayRet_t _aaValues(AA aa, size_t keysize, size_t valuesize)
    {
        std::string fname("_aaValues");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // void* _aaRehash(AA* paa, TypeInfo keyti)
    {
        std::string fname("_aaRehash");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(typeInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidPtrTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // ArrayRet_t _aaKeys(AA aa, size_t keysize)
    {
        std::string fname("_aaKeys");
        std::vector<const LLType*> types;
        types.push_back(rt_array(byteTy));
        types.push_back(aaTy);
        types.push_back(sizeTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // int _aaApply(AA aa, size_t keysize, dg_t dg)
    {
        std::string fname("_aaApply");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg1());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(3, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // int _aaApply2(AA aa, size_t keysize, dg2_t dg)
    {
        std::string fname("_aaApply2");
        std::vector<const LLType*> types;
        types.push_back(aaTy);
        types.push_back(sizeTy);
        types.push_back(rt_dg1());
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(3, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _moduleCtor()
    // void _moduleDtor()
    {
        std::string fname("_moduleCtor");
        std::string fname2("_moduleDtor");
        std::vector<const LLType*> types;
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Object _d_toObject(void* p)
    {
        std::string fname("_d_toObject");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_dynamic_cast(Object o, ClassInfo c)
    {
        std::string fname("_d_dynamic_cast");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    // Object _d_interface_cast(void* p, ClassInfo c)
    {
        std::string fname("_d_interface_cast");
        std::vector<const LLType*> types;
        types.push_back(voidPtrTy);
        types.push_back(classInfoTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(objectTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_throw_exception(Object e)
    {
        std::string fname("_d_throw_exception");
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
        std::string fname("_d_switch_string");
        std::vector<const LLType*> types;
        types.push_back(rt_array(rt_array2(byteTy)));
        types.push_back(stringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // int _d_switch_ustring(wchar[][] table, wchar[] ca)
    {
        std::string fname("_d_switch_ustring");
        std::vector<const LLType*> types;
        types.push_back(rt_array(rt_array2(shortTy)));
        types.push_back(wstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    // int _d_switch_dstring(dchar[][] table, dchar[] ca)
    {
        std::string fname("_d_switch_dstring");
        std::vector<const LLType*> types;
        types.push_back(rt_array(rt_array2(intTy)));
        types.push_back(dstringTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(intTy, types, false);
        llvm::AttrListPtr palist;
        palist = palist.addAttr(1, llvm::Attribute::ByVal);
        palist = palist.addAttr(2, llvm::Attribute::ByVal);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M)->setAttributes(palist);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // void _d_criticalenter(D_CRITICAL_SECTION *dcs)
    // void _d_criticalexit(D_CRITICAL_SECTION *dcs)
    {
        std::string fname("_d_criticalenter");
        std::string fname2("_d_criticalexit");
        std::vector<const LLType*> types;
        types.push_back(rt_ptr(DtoMutexType()));
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    // void _d_monitorenter(Object h)
    // void _d_monitorexit(Object h)
    {
        std::string fname("_d_monitorenter");
        std::string fname2("_d_monitorexit");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname2, M);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // int _d_eh_personality(int ver, int actions, ulong eh_class, ptr eh_info, ptr context)
    {
        std::string fname("_d_eh_personality");
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
        std::string fname("_d_eh_resume_unwind");
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
        std::string fname("_d_invariant");
        std::vector<const LLType*> types;
        types.push_back(objectTy);
        const llvm::FunctionType* fty = llvm::FunctionType::get(voidTy, types, false);
        llvm::Function::Create(fty, llvm::GlobalValue::ExternalLinkage, fname, M);
    }
}
