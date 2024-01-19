//===-- runtime.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/runtime.h"

#include "dmd/aggregate.h"
#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "dmd/tokens.h"
#include "driver/cl_options_instrumentation.h"
#include "gen/abi/abi.h"
#include "gen/attributes.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irtype.h"
#include "ir/irtypefunction.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Attributes.h"
#if LDC_LLVM_VER >= 1600
#include "llvm/Support/ModRef.h"
#endif
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool> nogc(
    "nogc", llvm::cl::ZeroOrMore,
    llvm::cl::desc(
        "Do not allow code that generates implicit garbage collector calls"));

////////////////////////////////////////////////////////////////////////////////

// Internal LLVM module containing already declared runtime functions and
// globals.
static llvm::Module *M = nullptr;

static void buildRuntimeModule();

////////////////////////////////////////////////////////////////////////////////

static void checkForImplicitGCCall(const Loc &loc, const char *name) {
  if (nogc) {
    static const std::string GCNAMES[] = {
        "_aaDelX",
        "_aaGetY",
        "_aaKeys",
        "_aaNew",
        "_aaRehash",
        "_aaValues",
        "_d_allocmemory",
        "_d_allocmemoryT",
        "_d_array_slice_copy",
        "_d_arrayappendT",
        "_d_arrayappendcTX",
        "_d_arrayappendcd",
        "_d_arrayappendwd",
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
        "_d_newarrayU",
        "_d_newclass",
        "_d_allocclass",
        // TODO: _d_newitemT and _d_newarraymTX instantiations
    };

    if (binary_search(&GCNAMES[0],
                      &GCNAMES[sizeof(GCNAMES) / sizeof(std::string)], name)) {
      error(loc,
            "No implicit garbage collector calls allowed with -nogc "
            "option enabled: `%s`",
            name);
      fatal();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

bool initRuntime() {
  if (!M) {
    Logger::println("*** Initializing D runtime declarations ***");
    LOG_SCOPE;

    buildRuntimeModule();
  }

  return true;
}

void freeRuntime() {
  if (M) {
    Logger::println("*** Freeing D runtime declarations ***");
    delete M;
    M = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Declaration> struct LazyType {
private:
  Declaration *&declRef;
  const char *const name;
  Type *type = nullptr;

  const char *getKind() { return "class"; }

public:
  LazyType(Declaration *&decl, const char *name) : declRef(decl), name(name) {}

  Type *get(const Loc &loc = {}) {
    if (!type) {
      if (!declRef || !declRef->type) {
        const char *kind = getKind();
        Logger::println("Missing %s declaration: %s\n", kind, name);
        error(loc, "Missing %s declaration: `%s`", kind, name);
        errorSupplemental(loc,
                          "Please check that object.d is included and valid");
        fatal();
      }
      type = declRef->type;
    }
    return type;
  }
};

using LazyClassType = LazyType<ClassDeclaration>;
LazyClassType objectTy(ClassDeclaration::object, "Object");
LazyClassType typeInfoTy(Type::dtypeinfo, "TypeInfo");
LazyClassType enumTypeInfoTy(Type::typeinfoenum, "TypeInfo_Enum");
LazyClassType pointerTypeInfoTy(Type::typeinfopointer, "TypeInfo_Pointer");
LazyClassType arrayTypeInfoTy(Type::typeinfoarray, "TypeInfo_Array");
LazyClassType staticArrayTypeInfoTy(Type::typeinfostaticarray,
                                    "TypeInfo_StaticArray");
LazyClassType aaTypeInfoTy(Type::typeinfoassociativearray,
                           "TypeInfo_AssociativeArray");
LazyClassType vectorTypeInfoTy(Type::typeinfovector, "TypeInfo_Vector");
LazyClassType functionTypeInfoTy(Type::typeinfofunction, "TypeInfo_Function");
LazyClassType delegateTypeInfoTy(Type::typeinfodelegate, "TypeInfo_Delegate");
LazyClassType classInfoTy(Type::typeinfoclass, "TypeInfo_Class");
LazyClassType interfaceTypeInfoTy(Type::typeinfointerface,
                                  "TypeInfo_Interface");
LazyClassType structTypeInfoTy(Type::typeinfostruct, "TypeInfo_Struct");
LazyClassType tupleTypeInfoTy(Type::typeinfotypelist, "TypeInfo_Tuple");
LazyClassType constTypeInfoTy(Type::typeinfoconst, "TypeInfo_Const");
LazyClassType invariantTypeInfoTy(Type::typeinfoinvariant,
                                  "TypeInfo_Invariant");
LazyClassType sharedTypeInfoTy(Type::typeinfoshared, "TypeInfo_Shared");
LazyClassType inoutTypeInfoTy(Type::typeinfowild, "TypeInfo_Inout");
LazyClassType throwableTy(ClassDeclaration::throwable, "Throwable");
LazyClassType cppTypeInfoPtrTy(ClassDeclaration::cpp_type_info_ptr,
                               "__cpp_type_info_ptr");

using LazyAggregateType = LazyType<AggregateDeclaration>;
template <> const char *LazyAggregateType::getKind() { return "struct"; }
LazyAggregateType moduleInfoTy(Module::moduleinfo, "ModuleInfo");

////////////////////////////////////////////////////////////////////////////////

struct PotentiallyLazyType {
private:
  enum class Kind {
    normal,
    lazyClass,
    lazyAggregate,
  };
  Kind kind;
  int numIndirections = 0;
  void *ptr;

public:
  PotentiallyLazyType(Type *type) : kind(Kind::normal), ptr(type) {}
  PotentiallyLazyType(LazyClassType &type)
      : kind(Kind::lazyClass), ptr(&type) {}
  PotentiallyLazyType(LazyAggregateType &type)
      : kind(Kind::lazyAggregate), ptr(&type) {}

  PotentiallyLazyType pointerTo() const {
    auto copy = *this;
    copy.numIndirections++;
    return copy;
  }

  Type *get(const Loc &loc) const {
    Type *ty;
    if (kind == Kind::lazyClass) {
      ty = static_cast<LazyClassType *>(ptr)->get(loc);
    } else if (kind == Kind::lazyAggregate) {
      ty = static_cast<LazyAggregateType *>(ptr)->get(loc);
    } else {
      ty = static_cast<Type *>(ptr);
    }

    for (int i = 0; i < numIndirections; ++i)
      ty = ty->pointerTo();

    return ty;
  }
};

const auto moduleInfoPtrTy = PotentiallyLazyType(moduleInfoTy).pointerTo();
const auto objectPtrTy = PotentiallyLazyType(objectTy).pointerTo();

////////////////////////////////////////////////////////////////////////////////

struct LazyFunctionDeclarer {
  LINK linkage;
  PotentiallyLazyType returnType;
  std::vector<llvm::StringRef> mangledFunctionNames;
  std::vector<PotentiallyLazyType> paramTypes;
  std::vector<StorageClass> paramsSTC;
  AttrSet attributes;

  void declare(const Loc &loc) {
    Parameters *params = nullptr;
    if (!paramTypes.empty()) {
      params = createParameters();
      for (size_t i = 0, e = paramTypes.size(); i < e; ++i) {
        StorageClass stc = paramsSTC.empty() ? 0 : paramsSTC[i];
        Type *paramTy = paramTypes[i].get(loc);
        params->push(
            Parameter::create(Loc(), stc, paramTy, nullptr, nullptr, nullptr));
      }
    }
    Type *returnTy = returnType.get(loc);
    auto dty = TypeFunction::create(params, returnTy, VARARGnone, linkage);

    // the call to DtoType performs many actions such as rewriting the function
    // type and storing it in dty
    auto llfunctype = llvm::cast<llvm::FunctionType>(DtoType(dty));
    auto attrs = getIrType(dty)->getIrFuncTy().getParamAttrs(
        gABI->passThisBeforeSret(dty));
    attrs.merge(attributes);

    for (auto fname : mangledFunctionNames) {
      llvm::Function *fn = llvm::Function::Create(
          llfunctype, llvm::GlobalValue::ExternalLinkage, fname, M);

      fn->setAttributes(attrs);

      // On x86_64, always set 'uwtable' for System V ABI compatibility.
      // FIXME: Move to better place (abi-x86-64.cpp?)
      // NOTE: There are several occurances if this line.
      if (global.params.targetTriple->getArch() == llvm::Triple::x86_64) {
#if LDC_LLVM_VER >= 1500
        fn->setUWTableKind(llvm::UWTableKind::Default);
#else
        fn->addFnAttr(LLAttribute::UWTable);
#endif
      }

      fn->setCallingConv(gABI->callingConv(dty, false));
    }
  }
};

// Use a pointer in order to share one declarer (declaring multiple functions
// of the same type under different names) for multiple function names.
llvm::StringMap<LazyFunctionDeclarer *> lazyFunctionDeclarers;

// Registers a runtime function forward declaration. The actual declaration of
// the function (and involved types) is deferred to the first
// getRuntimeFunction() call.
void createFwdDecl(LINK linkage, PotentiallyLazyType returnType,
                   std::vector<llvm::StringRef> mangledFunctionNames,
                   std::vector<PotentiallyLazyType> paramTypes,
                   std::vector<StorageClass> paramsSTC = {},
                   AttrSet attributes = {}) {
  const auto ptr = new LazyFunctionDeclarer{linkage,
                                            returnType,
                                            mangledFunctionNames,
                                            std::move(paramTypes),
                                            std::move(paramsSTC),
                                            attributes};

  for (auto name : mangledFunctionNames)
    lazyFunctionDeclarers[name] = ptr;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

llvm::Function *getRuntimeFunction(const Loc &loc, llvm::Module &target,
                                   const char *name) {
  checkForImplicitGCCall(loc, name);

  if (!M)
    initRuntime();

  LLFunction *fn = M->getFunction(name);
  if (!fn) {
    const auto it = lazyFunctionDeclarers.find(name);
    if (it == lazyFunctionDeclarers.end()) {
      error(loc, "Runtime function `%s` was not found", name);
      fatal();
    }
    // declare it in the M runtime module
    it->second->declare(loc);
    fn = M->getFunction(name);
    assert(fn);
  }
  LLFunctionType *fnty = fn->getFunctionType();

  if (LLFunction *existing = target.getFunction(name)) {
    if (existing->getFunctionType() != fnty) {
      error(Loc(), "Incompatible declaration of runtime function `%s`", name);
      fatal();
    }
    return existing;
  }

  LLFunction *resfn = llvm::cast<llvm::Function>(
      target.getOrInsertFunction(name, fnty).getCallee());
  resfn->setAttributes(fn->getAttributes());
  resfn->setCallingConv(fn->getCallingConv());
  return resfn;
}

////////////////////////////////////////////////////////////////////////////////

// C assert function:
// OSX:     void __assert_rtn(const char *func, const char *file, unsigned line,
//                            const char *msg)
// Android: void __assert(const char *file, int line, const char *msg)
// MSVC:    void  _assert(const char *msg, const char *file, unsigned line)
// Solaris: void __assert_c99(const char *assertion, const char *filename, int line_num,
//                            const char *funcname);
// Musl:    void __assert_fail(const char *assertion, const char *filename, int line_num,
//                             const char *funcname);
// uClibc:  void __assert(const char *assertion, const char *filename, int linenumber,
//                        const char *function);
// newlib:  void __assert_func(const char *file, int line, const char *func,
//                             const char *failedexpr)
// else:    void __assert(const char *msg, const char *file, unsigned line)

static const char *getCAssertFunctionName() {
  const auto &triple = *global.params.targetTriple;
  if (triple.isOSDarwin()) {
    return "__assert_rtn";
  } else if (triple.isWindowsMSVCEnvironment()) {
    return "_assert";
  } else if (triple.isOSSolaris()) {
    return "__assert_c99";
  } else if (triple.isMusl()) {
    return "__assert_fail";
  } else if (global.params.isNewlibEnvironment) {
    return "__assert_func";
  }
  return "__assert";
}

static std::vector<PotentiallyLazyType> getCAssertFunctionParamTypes() {
  const auto &triple = *global.params.targetTriple;
  const auto voidPtr = Type::tvoidptr;
  const auto uint = Type::tuns32;

  if (triple.isOSDarwin() || triple.isOSSolaris() || triple.isMusl() ||
      global.params.isUClibcEnvironment) {
    return {voidPtr, voidPtr, uint, voidPtr};
  }
  if (triple.getEnvironment() == llvm::Triple::Android) {
    return {voidPtr, uint, voidPtr};
  }
  if (global.params.isNewlibEnvironment) {
    return {voidPtr, uint, voidPtr, voidPtr};
  }
  return {voidPtr, voidPtr, uint};
}

llvm::Function *getCAssertFunction(const Loc &loc, llvm::Module &target) {
  return getRuntimeFunction(loc, target, getCAssertFunctionName());
}

////////////////////////////////////////////////////////////////////////////////

// Continue-unwinding function:
// ARM EABI: void _d_eh_resume_unwind(void*)
// ARM iOS:  void _Unwind_SjLj_Resume(void*)
// else:     void _Unwind_Resume(void*)

static const char *getUnwindResumeFunctionName() {
  const auto &triple = *global.params.targetTriple;
  if (triple.getArch() == llvm::Triple::arm)
    return triple.isOSDarwin() ? "_Unwind_SjLj_Resume" : "_d_eh_resume_unwind";
  return "_Unwind_Resume";
}

llvm::Function *getUnwindResumeFunction(const Loc &loc, llvm::Module &target) {
  return getRuntimeFunction(loc, target, getUnwindResumeFunctionName());
}

////////////////////////////////////////////////////////////////////////////////

Type *getObjectType() { return objectTy.get(); }
Type *getTypeInfoType() { return typeInfoTy.get(); }
Type *getEnumTypeInfoType() { return enumTypeInfoTy.get(); }
Type *getPointerTypeInfoType() { return pointerTypeInfoTy.get(); }
Type *getArrayTypeInfoType() { return arrayTypeInfoTy.get(); }
Type *getStaticArrayTypeInfoType() { return staticArrayTypeInfoTy.get(); }
Type *getAssociativeArrayTypeInfoType() { return aaTypeInfoTy.get(); }
Type *getVectorTypeInfoType() { return vectorTypeInfoTy.get(); }
Type *getFunctionTypeInfoType() { return functionTypeInfoTy.get(); }
Type *getDelegateTypeInfoType() { return delegateTypeInfoTy.get(); }
Type *getClassInfoType() { return classInfoTy.get(); }
Type *getInterfaceTypeInfoType() { return interfaceTypeInfoTy.get(); }
Type *getStructTypeInfoType() { return structTypeInfoTy.get(); }
Type *getTupleTypeInfoType() { return tupleTypeInfoTy.get(); }
Type *getConstTypeInfoType() { return constTypeInfoTy.get(); }
Type *getInvariantTypeInfoType() { return invariantTypeInfoTy.get(); }
Type *getSharedTypeInfoType() { return sharedTypeInfoTy.get(); }
Type *getInoutTypeInfoType() { return inoutTypeInfoTy.get(); }
Type *getThrowableType() { return throwableTy.get(); }
Type *getCppTypeInfoPtrType() { return cppTypeInfoPtrTy.get(); }
Type *getModuleInfoType() { return moduleInfoTy.get(); }

////////////////////////////////////////////////////////////////////////////////

// extern (D) alias dg_t = int delegate(void*);
static Type *rt_dg1() {
  static Type *dg_t = nullptr;
  if (dg_t)
    return dg_t;

  auto params = createParameters();
  params->push(
      Parameter::create(Loc(), 0, Type::tvoidptr, nullptr, nullptr, nullptr));
  auto fty = TypeFunction::create(params, Type::tint32, VARARGnone, LINK::d);
  dg_t = TypeDelegate::create(fty);
  return dg_t;
}

// extern (D) alias dg2_t = int delegate(void*, void*);
static Type *rt_dg2() {
  static Type *dg2_t = nullptr;
  if (dg2_t)
    return dg2_t;

  auto params = createParameters();
  params->push(
      Parameter::create(Loc(), 0, Type::tvoidptr, nullptr, nullptr, nullptr));
  params->push(
      Parameter::create(Loc(), 0, Type::tvoidptr, nullptr, nullptr, nullptr));
  auto fty = TypeFunction::create(params, Type::tint32, VARARGnone, LINK::d);
  dg2_t = TypeDelegate::create(fty);
  return dg2_t;
}

static void buildRuntimeModule() {
  Logger::println("building runtime module");
  auto &context = gIR->context();
  M = new llvm::Module("ldc internal runtime", context);

  Type *voidTy = Type::tvoid;
  Type *boolTy = Type::tbool;
  Type *ubyteTy = Type::tuns8;
  Type *intTy = Type::tint32;
  Type *uintTy = Type::tuns32;
  Type *ulongTy = Type::tuns64;
  Type *sizeTy = Type::tsize_t;
  Type *dcharTy = Type::tdchar;

  Type *voidPtrTy = Type::tvoidptr;
  Type *voidArrayTy = Type::tvoid->arrayOf();
  Type *voidArrayPtrTy = voidArrayTy->pointerTo();
  Type *stringTy = Type::tchar->arrayOf();
  Type *wstringTy = Type::twchar->arrayOf();
  Type *dstringTy = Type::tdchar->arrayOf();

  // LDC's AA type is rt.aaA.Impl*; use void* for the prototypes
  Type *aaTy = voidPtrTy;

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Construct some attribute lists used below (possibly multiple times)
  AttrSet NoAttrs,
      Attr_NoUnwind(NoAttrs, LLAttributeList::FunctionIndex,
                    llvm::Attribute::NoUnwind),
#if LDC_LLVM_VER >= 1600
      Attr_ReadOnly(llvm::AttributeList().addFnAttribute(
          context, llvm::Attribute::getWithMemoryEffects(
                       context, llvm::MemoryEffects::readOnly()))),
#else
      Attr_ReadOnly(NoAttrs, LLAttributeList::FunctionIndex,
                    llvm::Attribute::ReadOnly),
#endif
      Attr_Cold(NoAttrs, LLAttributeList::FunctionIndex, llvm::Attribute::Cold),
      Attr_Cold_NoReturn(Attr_Cold, LLAttributeList::FunctionIndex,
                         llvm::Attribute::NoReturn),
      Attr_Cold_NoReturn_NoUnwind(Attr_Cold_NoReturn,
                                  LLAttributeList::FunctionIndex,
                                  llvm::Attribute::NoUnwind),
      Attr_ReadOnly_NoUnwind(Attr_ReadOnly, LLAttributeList::FunctionIndex,
                             llvm::Attribute::NoUnwind),
      Attr_ReadOnly_1_NoCapture(Attr_ReadOnly, LLAttributeList::FirstArgIndex,
                                llvm::Attribute::NoCapture),
      Attr_ReadOnly_1_3_NoCapture(Attr_ReadOnly_1_NoCapture,
                                  LLAttributeList::FirstArgIndex + 2,
                                  llvm::Attribute::NoCapture),
      Attr_ReadOnly_NoUnwind_1_NoCapture(Attr_ReadOnly_1_NoCapture,
                                         LLAttributeList::FunctionIndex,
                                         llvm::Attribute::NoUnwind),
      Attr_ReadOnly_NoUnwind_1_2_NoCapture(Attr_ReadOnly_NoUnwind_1_NoCapture,
                                           LLAttributeList::FirstArgIndex + 1,
                                           llvm::Attribute::NoCapture),
      Attr_1_NoCapture(NoAttrs, LLAttributeList::FirstArgIndex,
                       llvm::Attribute::NoCapture),
      Attr_1_2_NoCapture(Attr_1_NoCapture, LLAttributeList::FirstArgIndex + 1,
                         llvm::Attribute::NoCapture),
      Attr_1_3_NoCapture(Attr_1_NoCapture, LLAttributeList::FirstArgIndex + 2,
                         llvm::Attribute::NoCapture),
      Attr_1_4_NoCapture(Attr_1_NoCapture, LLAttributeList::FirstArgIndex + 3,
                         llvm::Attribute::NoCapture);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void __cyg_profile_func_enter(void *callee, void *caller)
  // void __cyg_profile_func_exit(void *callee, void *caller)
  createFwdDecl(LINK::c, voidTy,
                {"__cyg_profile_func_exit", "__cyg_profile_func_enter"},
                {voidPtrTy, voidPtrTy}, {}, Attr_NoUnwind);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // C assert function
  createFwdDecl(LINK::c, Type::tvoid, {getCAssertFunctionName()},
                getCAssertFunctionParamTypes(), {},
                Attr_Cold_NoReturn_NoUnwind);

  // void _d_assert(string file, uint line)
  // void _d_arraybounds(string file, uint line)
  createFwdDecl(LINK::c, Type::tvoid, {"_d_assert", "_d_arraybounds"},
                {stringTy, uintTy}, {}, Attr_Cold_NoReturn);

  // void _d_assert_msg(string msg, string file, uint line)
  createFwdDecl(LINK::c, voidTy, {"_d_assert_msg"}, {stringTy, stringTy, uintTy},
                {}, Attr_Cold_NoReturn);

  // void _d_arraybounds_slice(string file, uint line, size_t lower,
  //                           size_t upper, size_t length)
  createFwdDecl(LINK::c, voidTy, {"_d_arraybounds_slice"},
                {stringTy, uintTy, sizeTy, sizeTy, sizeTy}, {},
                Attr_Cold_NoReturn);

  // void _d_arraybounds_index(string file, uint line, size_t index,
  //                           size_t length)
  createFwdDecl(LINK::c, voidTy, {"_d_arraybounds_index"},
                {stringTy, uintTy, sizeTy, sizeTy}, {}, Attr_Cold_NoReturn);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void* _d_allocmemory(size_t sz)
  createFwdDecl(LINK::c, voidPtrTy, {"_d_allocmemory"}, {sizeTy});

  // void* _d_allocmemoryT(TypeInfo ti)
  createFwdDecl(LINK::c, voidPtrTy, {"_d_allocmemoryT"}, {typeInfoTy});

  // void[] _d_newarrayT (const TypeInfo ti, size_t length)
  // void[] _d_newarrayiT(const TypeInfo ti, size_t length)
  // void[] _d_newarrayU (const TypeInfo ti, size_t length)
  createFwdDecl(LINK::c, voidArrayTy,
                {"_d_newarrayT", "_d_newarrayiT", "_d_newarrayU"},
                {typeInfoTy, sizeTy}, {STCconst, 0});

  // void[] _d_arrayappendcd(ref byte[] x, dchar c)
  // void[] _d_arrayappendwd(ref byte[] x, dchar c)
  createFwdDecl(LINK::c, voidArrayTy, {"_d_arrayappendcd", "_d_arrayappendwd"},
                {voidArrayTy, dcharTy}, {STCref, 0});

  // Object _d_newclass(const ClassInfo ci)
  // Object _d_allocclass(const ClassInfo ci)
  createFwdDecl(LINK::c, objectTy, {"_d_newclass", "_d_allocclass"},
                {classInfoTy}, {STCconst});

  // Throwable _d_newThrowable(const ClassInfo ci)
  createFwdDecl(LINK::c, throwableTy, {"_d_newThrowable"}, {classInfoTy},
                {STCconst});

  // void _d_delarray_t(void[]* p, const TypeInfo_Struct ti)
  createFwdDecl(LINK::c, voidTy, {"_d_delarray_t"},
                {voidArrayPtrTy, structTypeInfoTy}, {0, STCconst});

  // void _d_delmemory(void** p)
  // void _d_delinterface(void** p)
  createFwdDecl(LINK::c, voidTy, {"_d_delmemory", "_d_delinterface"},
                {voidPtrTy->pointerTo()});

  // void _d_callfinalizer(void* p)
  createFwdDecl(LINK::c, voidTy, {"_d_callfinalizer"}, {voidPtrTy});

  // D2: void _d_delclass(Object* p)
  createFwdDecl(LINK::c, voidTy, {"_d_delclass"}, {objectPtrTy});

  // void _d_delstruct(void** p, TypeInfo_Struct inf)
  createFwdDecl(LINK::c, voidTy, {"_d_delstruct"},
                {voidPtrTy->pointerTo(), structTypeInfoTy});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // array slice copy when assertions are on!
  // void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t
  // srclen, size_t elemsz)
  createFwdDecl(LINK::c, voidTy, {"_d_array_slice_copy"},
                {voidPtrTy, sizeTy, voidPtrTy, sizeTy, sizeTy}, {},
                Attr_1_3_NoCapture);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// int _aApplycd1(in char[] aa, dg_t dg)
// int _aApplyRcd1(in char[] aa, dg_t dg)
#define STR_APPLY1(TY, a, b)                                                   \
  {                                                                            \
    const char *fname1 = "_aApply" #a "1", *fname2 = "_aApply" #b "1",         \
               *fname3 = "_aApplyR" #a "1", *fname4 = "_aApplyR" #b "1";       \
    createFwdDecl(LINK::c, sizeTy, {fname1, fname2, fname3, fname4},             \
                  {TY, rt_dg1()});                                             \
  }
  STR_APPLY1(stringTy, cw, cd)
  STR_APPLY1(wstringTy, wc, wd)
  STR_APPLY1(dstringTy, dc, dw)
#undef STR_APPLY1

// int _aApplycd2(in char[] aa, dg2_t dg)
// int _aApplyRcd2(in char[] aa, dg2_t dg)
#define STR_APPLY2(TY, a, b)                                                   \
  {                                                                            \
    const char *fname1 = "_aApply" #a "2", *fname2 = "_aApply" #b "2",         \
               *fname3 = "_aApplyR" #a "2", *fname4 = "_aApplyR" #b "2";       \
    createFwdDecl(LINK::c, sizeTy, {fname1, fname2, fname3, fname4},             \
                  {TY, rt_dg2()});                                             \
  }
  STR_APPLY2(stringTy, cw, cd)
  STR_APPLY2(wstringTy, wc, wd)
  STR_APPLY2(dstringTy, dc, dw)
#undef STR_APPLY2

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void[] _d_arrayassign_l(TypeInfo ti, void[] src, void[] dst, void* ptmp)
  // void[] _d_arrayassign_r(TypeInfo ti, void[] src, void[] dst, void* ptmp)
  createFwdDecl(LINK::c, voidArrayTy, {"_d_arrayassign_l", "_d_arrayassign_r"},
                {typeInfoTy, voidArrayTy, voidArrayTy, voidPtrTy});

  // void* _d_arraysetassign(void* p, void* value, int count, TypeInfo ti)
  createFwdDecl(LINK::c, voidPtrTy, {"_d_arraysetassign"},
                {voidPtrTy, voidPtrTy, intTy, typeInfoTy});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // cast interface
  // void* _d_interface_cast(void* p, ClassInfo c)
  createFwdDecl(LINK::c, voidPtrTy, {"_d_interface_cast"},
                {voidPtrTy, classInfoTy}, {}, Attr_ReadOnly_NoUnwind);

  // dynamic cast
  // void* _d_dynamic_cast(Object o, ClassInfo c)
  createFwdDecl(LINK::c, voidPtrTy, {"_d_dynamic_cast"}, {objectTy, classInfoTy},
                {}, Attr_ReadOnly_NoUnwind);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // int _adEq2(void[] a1, void[] a2, TypeInfo ti)
  createFwdDecl(LINK::c, intTy, {"_adEq2"},
                {voidArrayTy, voidArrayTy, typeInfoTy}, {}, Attr_ReadOnly);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void* _aaGetY(AA* aa, const TypeInfo aati, in size_t valuesize,
  //               in void* pkey)
  createFwdDecl(LINK::c, voidPtrTy, {"_aaGetY"},
                {aaTy->pointerTo(), aaTypeInfoTy, sizeTy, voidPtrTy},
                {0, STCconst, STCin, STCin}, Attr_1_4_NoCapture);

  // inout(void)* _aaInX(inout AA aa, in TypeInfo keyti, in void* pkey)
  // FIXME: "inout" storageclass is not applied to return type
  createFwdDecl(LINK::c, voidPtrTy, {"_aaInX"}, {aaTy, typeInfoTy, voidPtrTy},
                {STCin | STCout, STCin, STCin}, Attr_ReadOnly_1_3_NoCapture);

  // bool _aaDelX(AA aa, in TypeInfo keyti, in void* pkey)
  createFwdDecl(LINK::c, boolTy, {"_aaDelX"}, {aaTy, typeInfoTy, voidPtrTy},
                {0, STCin, STCin}, Attr_1_3_NoCapture);

  // int _aaEqual(in TypeInfo tiRaw, in AA e1, in AA e2)
  createFwdDecl(LINK::c, intTy, {"_aaEqual"}, {typeInfoTy, aaTy, aaTy},
                {STCin, STCin, STCin}, Attr_1_2_NoCapture);

  // AA _d_assocarrayliteralTX(const TypeInfo_AssociativeArray ti,
  //                           void[] keys, void[] values)
  createFwdDecl(LINK::c, aaTy, {"_d_assocarrayliteralTX"},
                {aaTypeInfoTy, voidArrayTy, voidArrayTy}, {STCconst, 0, 0});

  // AA _aaNew(const TypeInfo_AssociativeArray ti)
  createFwdDecl(LINK::c, aaTy, {"_aaNew"}, {aaTypeInfoTy}, {STCconst});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void _d_throw_exception(Throwable o)
  createFwdDecl(LINK::c, voidTy, {"_d_throw_exception"}, {throwableTy}, {},
                Attr_Cold_NoReturn);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // int _d_eh_personality(...)
  {
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      const char *fname =
          useMSVCEH() ? "__CxxFrameHandler3" : "_d_eh_personality";
      // (ptr ExceptionRecord, ptr EstablisherFrame, ptr ContextRecord,
      //  ptr DispatcherContext)
      createFwdDecl(LINK::c, intTy, {fname},
                    {voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy});
    } else if (global.params.targetTriple->getArch() == llvm::Triple::arm) {
      // (int state, ptr ucb, ptr context)
      createFwdDecl(LINK::c, intTy, {"_d_eh_personality"},
                    {intTy, voidPtrTy, voidPtrTy});
    } else {
      // (int ver, int actions, ulong eh_class, ptr eh_info, ptr context)
      createFwdDecl(LINK::c, intTy, {"_d_eh_personality"},
                    {intTy, intTy, ulongTy, voidPtrTy, voidPtrTy});
    }
  }

  if (useMSVCEH()) {
    // _d_enter_cleanup(ptr frame)
    createFwdDecl(LINK::c, boolTy, {"_d_enter_cleanup"}, {voidPtrTy});

    // _d_leave_cleanup(ptr frame)
    createFwdDecl(LINK::c, voidTy, {"_d_leave_cleanup"}, {voidPtrTy});

    // Throwable _d_eh_enter_catch(ptr exception, ClassInfo catchType)
    createFwdDecl(LINK::c, throwableTy, {"_d_eh_enter_catch"},
                  {voidPtrTy, classInfoTy}, {});
  } else {
    // void _Unwind_Resume(ptr)
    createFwdDecl(LINK::c, voidTy, {getUnwindResumeFunctionName()}, {voidPtrTy},
                  {}, Attr_Cold_NoReturn);

    // Throwable _d_eh_enter_catch(ptr)
    createFwdDecl(LINK::c, throwableTy, {"_d_eh_enter_catch"}, {voidPtrTy}, {},
                  Attr_NoUnwind);

    // void* __cxa_begin_catch(ptr)
    createFwdDecl(LINK::c, voidPtrTy, {"__cxa_begin_catch"}, {voidPtrTy}, {},
                  Attr_NoUnwind);
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void invariant._d_invariant(Object o)
  {
    static const std::string mangle =
        getIRMangledFuncName("_D9invariant12_d_invariantFC6ObjectZv", LINK::d);
    createFwdDecl(LINK::d, voidTy, {mangle}, {objectTy});
  }

  // void _d_dso_registry(void* data)
  // (the argument is really a pointer to
  // rt.sections_elf_shared.CompilerDSOData)
  createFwdDecl(LINK::c, voidTy, {"_d_dso_registry"}, {voidPtrTy});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // extern (C) void _d_cover_register2(string filename, size_t[] valid,
  //                                    uint[] data, ubyte minPercent)
  if (global.params.cov) {
    createFwdDecl(LINK::c, voidTy, {"_d_cover_register2"},
                  {stringTy, sizeTy->arrayOf(), uintTy->arrayOf(), ubyteTy});
  }

  if (target.objc.supported) {
    assert(global.params.targetTriple->isOSDarwin());

    // The types of these functions don't really matter because they are always
    // bitcast to correct signature before calling.
    Type *objectPtrTy = voidPtrTy;
    Type *selectorPtrTy = voidPtrTy;
    Type *realTy = Type::tfloat80;

    // id objc_msgSend(id self, SEL op, ...)
    // Function called early and/or often, so lazy binding isn't worthwhile.
    createFwdDecl(LINK::c, objectPtrTy, {"objc_msgSend"},
                  {objectPtrTy, selectorPtrTy}, {},
                  AttrSet(NoAttrs, ~0U, llvm::Attribute::NonLazyBind));

    switch (global.params.targetTriple->getArch()) {
    case llvm::Triple::x86_64:
      // creal objc_msgSend_fp2ret(id self, SEL op, ...)
      createFwdDecl(LINK::c, Type::tcomplex80, {"objc_msgSend_fp2ret"},
                    {objectPtrTy, selectorPtrTy});
    // fall-thru
    case llvm::Triple::x86:
      // x86_64 real return only,  x86 float, double, real return
      // real objc_msgSend_fpret(id self, SEL op, ...)
      createFwdDecl(LINK::c, realTy, {"objc_msgSend_fpret"},
                    {objectPtrTy, selectorPtrTy});
    // fall-thru
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // used when return value is aggregate via a hidden sret arg
      // void objc_msgSend_stret(T *sret_arg, id self, SEL op, ...)
      createFwdDecl(LINK::c, voidTy, {"objc_msgSend_stret"},
                    {objectPtrTy, selectorPtrTy});
      break;
    default:
      break;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// DMD-style tracing calls

  // extern(C) void trace_pro(char[] id)
  createFwdDecl(LINK::c, voidTy, {"trace_pro"}, {stringTy});

  // extern(C) void _c_trace_epi()
  createFwdDecl(LINK::c, voidTy, {"_c_trace_epi"}, {});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// C standard library functions (a druntime link dependency)

  // int memcmp(const void *s1, const void *s2, size_t n);
  createFwdDecl(LINK::c, intTy, {"memcmp"}, {voidPtrTy, voidPtrTy, sizeTy}, {},
                Attr_ReadOnly_NoUnwind_1_2_NoCapture);
}

static void emitInstrumentationFn(const char *name) {
  LLFunction *fn = getRuntimeFunction(Loc(), gIR->module, name);

  // Grab the address of the calling function
  auto *caller =
      gIR->ir->CreateCall(GET_INTRINSIC_DECL(returnaddress), DtoConstInt(0));
  auto callee = DtoBitCast(gIR->topfunc(), getVoidPtrType());

  gIR->ir->CreateCall(fn, {callee, caller});
}

void emitInstrumentationFnEnter(FuncDeclaration *decl) {
  if (opts::instrumentFunctions && decl->emitInstrumentation)
    emitInstrumentationFn("__cyg_profile_func_enter");
}

void emitInstrumentationFnLeave(FuncDeclaration *decl) {
  if (opts::instrumentFunctions && decl->emitInstrumentation)
    emitInstrumentationFn("__cyg_profile_func_exit");
}
