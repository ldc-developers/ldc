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
#include "gen/abi.h"
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
#include "driver/cl_options_instrumentation.h"
#include "ldcbindings.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "root.h"
#include "tokens.h"
#if LDC_LLVM_VER >= 400
#include "llvm/Bitcode/BitcodeWriter.h"
#else
#include "llvm/Bitcode/ReaderWriter.h"
#endif
#include "llvm/IR/Attributes.h"
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

// Internal LLVM module containing runtime declarations (functions and globals)
static llvm::Module *M = nullptr;

static void buildRuntimeModule();

////////////////////////////////////////////////////////////////////////////////

static void checkForImplicitGCCall(const Loc &loc, const char *name) {
  if (nogc) {
    static const std::string GCNAMES[] = {
        "_aaDelX",
        "_aaGetY",
        "_aaKeys",
        "_aaRehash",
        "_aaValues",
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
        "_d_allocclass",
        "_d_newitemT",
        "_d_newitemiT",
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
  Logger::println("*** Initializing D runtime declarations ***");
  LOG_SCOPE;

  if (!M) {
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

llvm::Function *getRuntimeFunction(const Loc &loc, llvm::Module &target,
                                   const char *name) {
  checkForImplicitGCCall(loc, name);

  if (!M)
    initRuntime();

  LLFunction *fn = M->getFunction(name);
  if (!fn) {
    error(loc, "Runtime function `%s` was not found", name);
    fatal();
  }
  LLFunctionType *fnty = fn->getFunctionType();

  if (LLFunction *existing = target.getFunction(name)) {
    if (existing->getFunctionType() != fnty) {
      error(Loc(), "Incompatible declaration of runtime function `%s`", name);
      fatal();
    }
    return existing;
  }

  LLFunction *resfn =
      llvm::cast<llvm::Function>(target.getOrInsertFunction(name, fnty));
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
// else:    void __assert(const char *msg, const char *file, unsigned line)

static const char *getCAssertFunctionName() {
  if (global.params.targetTriple->isOSDarwin()) {
    return "__assert_rtn";
  } else if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
    return "_assert";
  } else if (global.params.targetTriple->isOSSolaris()) {
    return "__assert_c99";
  }
  return "__assert";
}

static std::vector<Type *> getCAssertFunctionParamTypes() {
  const auto voidPtr = Type::tvoidptr;
  const auto uint = Type::tuns32;

  if (global.params.targetTriple->isOSDarwin() || global.params.targetTriple->isOSSolaris()) {
    return {voidPtr, voidPtr, uint, voidPtr};
  }
  if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
    return {voidPtr, uint, voidPtr};
  }
  return {voidPtr, voidPtr, uint};
}

llvm::Function *getCAssertFunction(const Loc &loc, llvm::Module &target) {
  return getRuntimeFunction(loc, target, getCAssertFunctionName());
}

////////////////////////////////////////////////////////////////////////////////

// extern (D) alias dg_t = int delegate(void*);
static Type *rt_dg1() {
  static Type *dg_t = nullptr;
  if (dg_t)
    return dg_t;

  auto params = new Parameters();
  params->push(Parameter::create(0, Type::tvoidptr, nullptr, nullptr));
  auto fty = TypeFunction::create(params, Type::tint32, 0, LINKd);
  dg_t = createTypeDelegate(fty);
  return dg_t;
}

// extern (D) alias dg2_t = int delegate(void*, void*);
static Type *rt_dg2() {
  static Type *dg2_t = nullptr;
  if (dg2_t)
    return dg2_t;

  auto params = new Parameters();
  params->push(Parameter::create(0, Type::tvoidptr, nullptr, nullptr));
  params->push(Parameter::create(0, Type::tvoidptr, nullptr, nullptr));
  auto fty = TypeFunction::create(params, Type::tint32, 0, LINKd);
  dg2_t = createTypeDelegate(fty);
  return dg2_t;
}

template <typename DECL> static void ensureDecl(DECL *decl, const char *msg) {
  if (!decl || !decl->type) {
    Logger::println("Missing class declaration: %s\n", msg);
    error(Loc(), "Missing class declaration: `%s`", msg);
    errorSupplemental(Loc(),
                      "Please check that object.d is included and valid");
    fatal();
  }
}

// Parameters fnames are assumed to be already mangled!
static void createFwdDecl(LINK linkage, Type *returntype,
                          ArrayParam<llvm::StringRef> fnames,
                          ArrayParam<Type *> paramtypes,
                          ArrayParam<StorageClass> paramsSTC = {},
                          AttrSet attribset = AttrSet()) {

  Parameters *params = nullptr;
  if (!paramtypes.empty()) {
    params = new Parameters();
    for (size_t i = 0, e = paramtypes.size(); i < e; ++i) {
      StorageClass stc = paramsSTC.empty() ? 0 : paramsSTC[i];
      params->push(Parameter::create(stc, paramtypes[i], nullptr, nullptr));
    }
  }
  int varargs = 0;
  auto dty = TypeFunction::create(params, returntype, varargs, linkage);

  // the call to DtoType performs many actions such as rewriting the function
  // type and storing it in dty
  auto llfunctype = llvm::cast<llvm::FunctionType>(DtoType(dty));
  assert(dty->ctype);
  auto attrs =
      dty->ctype->getIrFuncTy().getParamAttrs(gABI->passThisBeforeSret(dty));
  attrs.merge(attribset);

  for (auto fname : fnames) {
    llvm::Function *fn = llvm::Function::Create(
        llfunctype, llvm::GlobalValue::ExternalLinkage, fname, M);

    fn->setAttributes(attrs);

    // On x86_64, always set 'uwtable' for System V ABI compatibility.
    // FIXME: Move to better place (abi-x86-64.cpp?)
    // NOTE: There are several occurances if this line.
    if (global.params.targetTriple->getArch() == llvm::Triple::x86_64) {
      fn->addFnAttr(LLAttribute::UWTable);
    }

    fn->setCallingConv(gABI->callingConv(linkage, dty));
  }
}

static void buildRuntimeModule() {
  Logger::println("building runtime module");
  M = new llvm::Module("ldc internal runtime", gIR->context());

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

  // Ensure that the declarations exist before creating llvm types for them.
  ensureDecl(ClassDeclaration::object, "Object");
  ensureDecl(Type::typeinfoclass, "TypeInfo_Class");
  ensureDecl(Type::dtypeinfo, "TypeInfo");
  ensureDecl(Type::typeinfoassociativearray, "TypeInfo_AssociativeArray");
  ensureDecl(Module::moduleinfo, "ModuleInfo");

  Type *objectTy = ClassDeclaration::object->type;
  Type *throwableTy = ClassDeclaration::throwable->type;
  Type *classInfoTy = Type::typeinfoclass->type;
  Type *typeInfoTy = Type::dtypeinfo->type;
  Type *aaTypeInfoTy = Type::typeinfoassociativearray->type;
  Type *moduleInfoPtrTy = Module::moduleinfo->type->pointerTo();
  // The AA type is a struct that only contains a ptr
  Type *aaTy = voidPtrTy;

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Construct some attribute lists used below (possibly multiple times)
  AttrSet NoAttrs,
      Attr_NoAlias(NoAttrs, LLAttributeSet::ReturnIndex,
                   llvm::Attribute::NoAlias),
      Attr_NoUnwind(NoAttrs, LLAttributeSet::FunctionIndex,
                    llvm::Attribute::NoUnwind),
      Attr_ReadOnly(NoAttrs, LLAttributeSet::FunctionIndex,
                    llvm::Attribute::ReadOnly),
      Attr_Cold(NoAttrs, LLAttributeSet::FunctionIndex, llvm::Attribute::Cold),
      Attr_Cold_NoReturn(Attr_Cold, LLAttributeSet::FunctionIndex,
                         llvm::Attribute::NoReturn),
      Attr_Cold_NoReturn_NoUnwind(Attr_Cold_NoReturn,
                                  LLAttributeSet::FunctionIndex,
                                  llvm::Attribute::NoUnwind),
      Attr_ReadOnly_NoUnwind(Attr_ReadOnly, LLAttributeSet::FunctionIndex,
                             llvm::Attribute::NoUnwind),
      Attr_ReadOnly_1_NoCapture(Attr_ReadOnly, AttrSet::FirstArgIndex,
                                llvm::Attribute::NoCapture),
      Attr_ReadOnly_1_3_NoCapture(Attr_ReadOnly_1_NoCapture,
                                  AttrSet::FirstArgIndex + 2,
                                  llvm::Attribute::NoCapture),
      Attr_ReadOnly_NoUnwind_1_NoCapture(Attr_ReadOnly_1_NoCapture,
                                         LLAttributeSet::FunctionIndex,
                                         llvm::Attribute::NoUnwind),
      Attr_ReadOnly_NoUnwind_1_2_NoCapture(Attr_ReadOnly_NoUnwind_1_NoCapture,
                                           AttrSet::FirstArgIndex + 1,
                                           llvm::Attribute::NoCapture),
      Attr_ReadNone(NoAttrs, LLAttributeSet::FunctionIndex,
                    llvm::Attribute::ReadNone),
      Attr_1_NoCapture(NoAttrs, AttrSet::FirstArgIndex,
                       llvm::Attribute::NoCapture),
      Attr_1_2_NoCapture(Attr_1_NoCapture, AttrSet::FirstArgIndex + 1,
                         llvm::Attribute::NoCapture),
      Attr_1_3_NoCapture(Attr_1_NoCapture, AttrSet::FirstArgIndex + 2,
                         llvm::Attribute::NoCapture),
      Attr_1_4_NoCapture(Attr_1_NoCapture, AttrSet::FirstArgIndex + 3,
                         llvm::Attribute::NoCapture),
      Attr_NoAlias_1_NoCapture(Attr_1_NoCapture, LLAttributeSet::ReturnIndex,
                               llvm::Attribute::NoAlias);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void __cyg_profile_func_enter(void *callee, void *caller)
  // void __cyg_profile_func_exit(void *callee, void *caller)
  createFwdDecl(LINKc, voidTy,
                {"__cyg_profile_func_exit", "__cyg_profile_func_enter"},
                {voidPtrTy, voidPtrTy}, {}, Attr_NoUnwind);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // C assert function
  createFwdDecl(LINKc, Type::tvoid, {getCAssertFunctionName()},
                getCAssertFunctionParamTypes(), {},
                Attr_Cold_NoReturn_NoUnwind);

  // void _d_assert(string file, uint line)
  // void _d_arraybounds(string file, uint line)
  createFwdDecl(LINKc, Type::tvoid, {"_d_assert", "_d_arraybounds"},
                {stringTy, uintTy}, {}, Attr_Cold_NoReturn);

  // void _d_assert_msg(string msg, string file, uint line)
  createFwdDecl(LINKc, voidTy, {"_d_assert_msg"}, {stringTy, stringTy, uintTy},
                {}, Attr_Cold_NoReturn);

  // void _d_switch_error(immutable(ModuleInfo)* m, uint line)
  createFwdDecl(LINKc, voidTy, {"_d_switch_error"}, {moduleInfoPtrTy, uintTy},
                {STCimmutable, 0}, Attr_Cold_NoReturn);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void* _d_allocmemory(size_t sz)
  createFwdDecl(LINKc, voidPtrTy, {"_d_allocmemory"}, {sizeTy}, {},
                Attr_NoAlias);

  // void* _d_allocmemoryT(TypeInfo ti)
  createFwdDecl(LINKc, voidPtrTy, {"_d_allocmemoryT"}, {typeInfoTy}, {},
                Attr_NoAlias);

  // void[] _d_newarrayT (const TypeInfo ti, size_t length)
  // void[] _d_newarrayiT(const TypeInfo ti, size_t length)
  // void[] _d_newarrayU (const TypeInfo ti, size_t length)
  createFwdDecl(LINKc, voidArrayTy,
                {"_d_newarrayT", "_d_newarrayiT", "_d_newarrayU"},
                {typeInfoTy, sizeTy}, {STCconst, 0});

  // void[] _d_newarraymTX (const TypeInfo ti, size_t[] dims)
  // void[] _d_newarraymiTX(const TypeInfo ti, size_t[] dims)
  createFwdDecl(LINKc, voidArrayTy, {"_d_newarraymTX", "_d_newarraymiTX"},
                {typeInfoTy, sizeTy->arrayOf()}, {STCconst, 0});

  // void[] _d_arraysetlengthT (const TypeInfo ti, size_t newlength, void[]* p)
  // void[] _d_arraysetlengthiT(const TypeInfo ti, size_t newlength, void[]* p)
  createFwdDecl(LINKc, voidArrayTy,
                {"_d_arraysetlengthT", "_d_arraysetlengthiT"},
                {typeInfoTy, sizeTy, voidArrayPtrTy}, {STCconst, 0, 0});

  // byte[] _d_arrayappendcTX(const TypeInfo ti, ref byte[] px, size_t n)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arrayappendcTX"},
                {typeInfoTy, voidArrayTy, sizeTy}, {STCconst, STCref, 0});

  // void[] _d_arrayappendT(const TypeInfo ti, ref byte[] x, byte[] y)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arrayappendT"},
                {typeInfoTy, voidArrayTy, voidArrayTy}, {STCconst, STCref, 0});

  // void[] _d_arrayappendcd(ref byte[] x, dchar c)
  // void[] _d_arrayappendwd(ref byte[] x, dchar c)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arrayappendcd", "_d_arrayappendwd"},
                {voidArrayTy, dcharTy}, {STCref, 0});

  // byte[] _d_arraycatT(const TypeInfo ti, byte[] x, byte[] y)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arraycatT"},
                {typeInfoTy, voidArrayTy, voidArrayTy}, {STCconst, 0, 0});

  // void[] _d_arraycatnTX(const TypeInfo ti, byte[][] arrs)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arraycatnTX"},
                {typeInfoTy, voidArrayTy->arrayOf()}, {STCconst, 0});

  // Object _d_newclass(const ClassInfo ci)
  // Object _d_allocclass(const ClassInfo ci)
  createFwdDecl(LINKc, objectTy, {"_d_newclass", "_d_allocclass"},
                {classInfoTy}, {STCconst}, Attr_NoAlias);

  // void* _d_newitemT (TypeInfo ti)
  // void* _d_newitemiT(TypeInfo ti)
  createFwdDecl(LINKc, voidPtrTy, {"_d_newitemT", "_d_newitemiT"}, {typeInfoTy},
                {0}, Attr_NoAlias);

  // void _d_delarray_t(void[]* p, const TypeInfo_Struct ti)
  createFwdDecl(LINKc, voidTy, {"_d_delarray_t"},
                {voidArrayPtrTy, Type::typeinfostruct->type}, {0, STCconst});

  // void _d_delmemory(void** p)
  // void _d_delinterface(void** p)
  createFwdDecl(LINKc, voidTy, {"_d_delmemory", "_d_delinterface"},
                {voidPtrTy->pointerTo()});

  // void _d_callfinalizer(void* p)
  createFwdDecl(LINKc, voidTy, {"_d_callfinalizer"}, {voidPtrTy});

  // D2: void _d_delclass(Object* p)
  createFwdDecl(LINKc, voidTy, {"_d_delclass"}, {objectTy->pointerTo()});

  // void _d_delstruct(void** p, TypeInfo_Struct inf)
  createFwdDecl(LINKc, voidTy, {"_d_delstruct"},
                {voidPtrTy->pointerTo(), Type::typeinfostruct->type});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // array slice copy when assertions are on!
  // void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t
  // srclen)
  createFwdDecl(LINKc, voidTy, {"_d_array_slice_copy"},
                {voidPtrTy, sizeTy, voidPtrTy, sizeTy}, {}, Attr_1_3_NoCapture);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// int _aApplycd1(in char[] aa, dg_t dg)
// int _aApplyRcd1(in char[] aa, dg_t dg)
#define STR_APPLY1(TY, a, b)                                                   \
  {                                                                            \
    const std::string prefix = "_aApply";                                      \
    std::string fname1 = prefix + (a) + '1', fname2 = prefix + (b) + '1',      \
                fname3 = prefix + 'R' + (a) + '1',                             \
                fname4 = prefix + 'R' + (b) + '1';                             \
    createFwdDecl(LINKc, sizeTy, {fname1, fname2, fname3, fname4},             \
                  {TY, rt_dg1()});                                             \
  }
  STR_APPLY1(stringTy, "cw", "cd")
  STR_APPLY1(wstringTy, "wc", "wd")
  STR_APPLY1(dstringTy, "dc", "dw")
#undef STR_APPLY1

// int _aApplycd2(in char[] aa, dg2_t dg)
// int _aApplyRcd2(in char[] aa, dg2_t dg)
#define STR_APPLY2(TY, a, b)                                                   \
  {                                                                            \
    const std::string prefix = "_aApply";                                      \
    std::string fname1 = prefix + (a) + '2', fname2 = prefix + (b) + '2',      \
                fname3 = prefix + 'R' + (a) + '2',                             \
                fname4 = prefix + 'R' + (b) + '2';                             \
    createFwdDecl(LINKc, sizeTy, {fname1, fname2, fname3, fname4},             \
                  {TY, rt_dg2()});                                             \
  }
  STR_APPLY2(stringTy, "cw", "cd")
  STR_APPLY2(wstringTy, "wc", "wd")
  STR_APPLY2(dstringTy, "dc", "dw")
#undef STR_APPLY2

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // fixes the length for dynamic array casts
  // size_t _d_array_cast_len(size_t len, size_t elemsz, size_t newelemsz)
  createFwdDecl(LINKc, sizeTy, {"_d_array_cast_len"}, {sizeTy, sizeTy, sizeTy},
                {}, Attr_ReadNone);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void[] _d_arrayassign_l(TypeInfo ti, void[] src, void[] dst, void* ptmp)
  // void[] _d_arrayassign_r(TypeInfo ti, void[] src, void[] dst, void* ptmp)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arrayassign_l", "_d_arrayassign_r"},
                {typeInfoTy, voidArrayTy, voidArrayTy, voidPtrTy});

  // void[] _d_arrayctor(TypeInfo ti, void[] from, void[] to)
  createFwdDecl(LINKc, voidArrayTy, {"_d_arrayctor"},
                {typeInfoTy, voidArrayTy, voidArrayTy});

  // void* _d_arraysetassign(void* p, void* value, int count, TypeInfo ti)
  // void* _d_arraysetctor(void* p, void* value, int count, TypeInfo ti)
  createFwdDecl(LINKc, voidPtrTy, {"_d_arraysetassign", "_d_arraysetctor"},
                {voidPtrTy, voidPtrTy, intTy, typeInfoTy}, {}, Attr_NoAlias);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // cast interface
  // void* _d_interface_cast(void* p, ClassInfo c)
  createFwdDecl(LINKc, voidPtrTy, {"_d_interface_cast"},
                {voidPtrTy, classInfoTy}, {}, Attr_ReadOnly_NoUnwind);

  // dynamic cast
  // void* _d_dynamic_cast(Object o, ClassInfo c)
  createFwdDecl(LINKc, voidPtrTy, {"_d_dynamic_cast"}, {objectTy, classInfoTy},
                {}, Attr_ReadOnly_NoUnwind);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // int _adEq2(void[] a1, void[] a2, TypeInfo ti)
  createFwdDecl(LINKc, intTy, {"_adEq2"},
                {voidArrayTy, voidArrayTy, typeInfoTy}, {}, Attr_ReadOnly);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void* _aaGetY(AA* aa, const TypeInfo aati, in size_t valuesize,
  //               in void* pkey)
  createFwdDecl(LINKc, voidPtrTy, {"_aaGetY"},
                {aaTy->pointerTo(), aaTypeInfoTy, sizeTy, voidPtrTy},
                {0, STCconst, STCin, STCin}, Attr_1_4_NoCapture);

  // inout(void)* _aaInX(inout AA aa, in TypeInfo keyti, in void* pkey)
  // FIXME: "inout" storageclass is not applied to return type
  createFwdDecl(LINKc, voidPtrTy, {"_aaInX"}, {aaTy, typeInfoTy, voidPtrTy},
                {STCin | STCout, STCin, STCin}, Attr_ReadOnly_1_3_NoCapture);

  // bool _aaDelX(AA aa, in TypeInfo keyti, in void* pkey)
  createFwdDecl(LINKc, boolTy, {"_aaDelX"}, {aaTy, typeInfoTy, voidPtrTy},
                {0, STCin, STCin}, Attr_1_3_NoCapture);

  // int _aaEqual(in TypeInfo tiRaw, in AA e1, in AA e2)
  createFwdDecl(LINKc, intTy, {"_aaEqual"}, {typeInfoTy, aaTy, aaTy},
                {STCin, STCin, STCin}, Attr_1_2_NoCapture);

  // AA _d_assocarrayliteralTX(const TypeInfo_AssociativeArray ti,
  //                           void[] keys, void[] values)
  createFwdDecl(LINKc, aaTy, {"_d_assocarrayliteralTX"},
                {aaTypeInfoTy, voidArrayTy, voidArrayTy}, {STCconst, 0, 0});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void _d_throw_exception(Throwable o)
  createFwdDecl(LINKc, voidTy, {"_d_throw_exception"}, {throwableTy}, {},
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
      createFwdDecl(LINKc, intTy, {fname},
                    {voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy});
    } else if (global.params.targetTriple->getArch() == llvm::Triple::arm) {
      // (int state, ptr ucb, ptr context)
      createFwdDecl(LINKc, intTy, {"_d_eh_personality"},
                    {intTy, voidPtrTy, voidPtrTy});
    } else {
      // (int ver, int actions, ulong eh_class, ptr eh_info, ptr context)
      createFwdDecl(LINKc, intTy, {"_d_eh_personality"},
                    {intTy, intTy, ulongTy, voidPtrTy, voidPtrTy});
    }
  }

  if (useMSVCEH()) {
    // _d_enter_cleanup(ptr frame)
    createFwdDecl(LINKc, boolTy, {"_d_enter_cleanup"}, {voidPtrTy});

    // _d_leave_cleanup(ptr frame)
    createFwdDecl(LINKc, voidTy, {"_d_leave_cleanup"}, {voidPtrTy});

    // Throwable _d_eh_enter_catch(ptr exception, ClassInfo catchType)
    createFwdDecl(LINKc, throwableTy, {"_d_eh_enter_catch"},
                  {voidPtrTy, classInfoTy}, {});
  } else {
    // void _d_eh_resume_unwind(ptr)
    createFwdDecl(LINKc, voidTy, {"_d_eh_resume_unwind"}, {voidPtrTy});

    // Throwable _d_eh_enter_catch(ptr)
    createFwdDecl(LINKc, throwableTy, {"_d_eh_enter_catch"}, {voidPtrTy}, {},
                  Attr_NoUnwind);

    // void* __cxa_begin_catch(ptr)
    createFwdDecl(LINKc, voidPtrTy, {"__cxa_begin_catch"}, {voidPtrTy}, {},
                  Attr_NoUnwind);
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // void invariant._d_invariant(Object o)
  createFwdDecl(
      LINKd, voidTy,
      {getIRMangledFuncName("_D9invariant12_d_invariantFC6ObjectZv", LINKd)},
      {objectTy});

  // void _d_dso_registry(void* data)
  // (the argument is really a pointer to
  // rt.sections_elf_shared.CompilerDSOData)
  createFwdDecl(LINKc, voidTy, {"_d_dso_registry"}, {voidPtrTy});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // extern (C) void _d_cover_register2(string filename, size_t[] valid,
  //                                    uint[] data, ubyte minPercent)
  if (global.params.cov) {
    createFwdDecl(LINKc, voidTy, {"_d_cover_register2"},
                  {stringTy, sizeTy->arrayOf(), uintTy->arrayOf(), ubyteTy});
  }

  if (global.params.hasObjectiveC) {
    assert(global.params.targetTriple->isOSDarwin());

    // The types of these functions don't really matter because they are always
    // bitcast to correct signature before calling.
    Type *objectPtrTy = voidPtrTy;
    Type *selectorPtrTy = voidPtrTy;
    Type *realTy = Type::tfloat80;

    // id objc_msgSend(id self, SEL op, ...)
    // Function called early and/or often, so lazy binding isn't worthwhile.
    createFwdDecl(LINKc, objectPtrTy, {"objc_msgSend"},
                  {objectPtrTy, selectorPtrTy}, {},
                  AttrSet(NoAttrs, ~0U, llvm::Attribute::NonLazyBind));

    switch (global.params.targetTriple->getArch()) {
    case llvm::Triple::x86_64:
      // creal objc_msgSend_fp2ret(id self, SEL op, ...)
      createFwdDecl(LINKc, Type::tcomplex80, {"objc_msgSend_fp2ret"},
                    {objectPtrTy, selectorPtrTy});
    // fall-thru
    case llvm::Triple::x86:
      // x86_64 real return only,  x86 float, double, real return
      // real objc_msgSend_fpret(id self, SEL op, ...)
      createFwdDecl(LINKc, realTy, {"objc_msgSend_fpret"},
                    {objectPtrTy, selectorPtrTy});
    // fall-thru
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // used when return value is aggregate via a hidden sret arg
      // void objc_msgSend_stret(T *sret_arg, id self, SEL op, ...)
      createFwdDecl(LINKc, voidTy, {"objc_msgSend_stret"},
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
  createFwdDecl(LINKc, voidTy, {"trace_pro"}, {stringTy});

  // extern(C) void _c_trace_epi()
  createFwdDecl(LINKc, voidTy, {"_c_trace_epi"}, {});

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// C standard library functions (a druntime link dependency)

  // int memcmp(const void *s1, const void *s2, size_t n);
  createFwdDecl(LINKc, intTy, {"memcmp"}, {voidPtrTy, voidPtrTy, sizeTy}, {},
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
