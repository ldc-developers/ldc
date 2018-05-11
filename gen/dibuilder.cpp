//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dibuilder.h"

#include "driver/cl_options.h"
#include "driver/ldc-version.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/optimizer.h"
#include "ir/irfunction.h"
#include "ir/irtypeaggr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "enum.h"
#include "ldcbindings.h"
#include "module.h"
#include "mtype.h"

////////////////////////////////////////////////////////////////////////////////

using LLMetadata = llvm::Metadata;
using DIFlags = llvm::DINode;

namespace {
#if LDC_LLVM_VER >= 400
const auto DIFlagZero = DIFlags::FlagZero;
#else
const unsigned DIFlagZero = 0;
#endif

ldc::DIType getNullDIType() {
  return nullptr;
}

llvm::DINodeArray getEmptyDINodeArray() {
  return nullptr;
}

llvm::StringRef uniqueIdent(Type* t) {
#if LDC_LLVM_VER >= 309
  if (t->deco)
    return t->deco;
#endif
  return llvm::StringRef();
}

} // namespace


bool ldc::DIBuilder::mustEmitFullDebugInfo() {
  // only for -g and -gc 
  // TODO: but not dcompute (yet)

  if (IR->dcomputetarget) return false;

  return global.params.symdebug == 1 || global.params.symdebug == 2;
}

bool ldc::DIBuilder::mustEmitLocationsDebugInfo() {
  // for -g -gc and -gline-tables-only 
  // TODO:but not dcompute (yet)
    
  if (IR->dcomputetarget) return false;

  return (global.params.symdebug > 0) || global.params.outputSourceLocations;
}

////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current
// module
Module *ldc::DIBuilder::getDefinedModule(Dsymbol *s) {
  // templates are defined in current module
  if (DtoIsTemplateInstance(s)) {
    return IR->dmodule;
  }
  // otherwise use the symbol's module
  return s->getModule();
}

////////////////////////////////////////////////////////////////////////////////

ldc::DIBuilder::DIBuilder(IRState *const IR)
    : IR(IR), DBuilder(IR->module), CUNode(nullptr),
      isTargetMSVCx64(global.params.targetTriple->isWindowsMSVCEnvironment() &&
                      global.params.targetTriple->isArch64Bit()) {}

llvm::LLVMContext &ldc::DIBuilder::getContext() { return IR->context(); }

ldc::DIScope ldc::DIBuilder::GetCurrentScope() {
  IrFunction *fn = IR->func();
  if (fn->diLexicalBlocks.empty()) {
    assert(static_cast<llvm::MDNode *>(fn->diSubprogram) != 0);
    return fn->diSubprogram;
  }
  return fn->diLexicalBlocks.top();
}

// Sets the memory address for a debuginfo variable.
void ldc::DIBuilder::Declare(const Loc &loc, llvm::Value *storage,
                             ldc::DILocalVariable divar,
                             ldc::DIExpression diexpr) {
  unsigned charnum = (loc.linnum ? loc.charnum : 0);
  auto debugLoc = llvm::DebugLoc::get(loc.linnum, charnum, GetCurrentScope());
  DBuilder.insertDeclare(storage, divar, diexpr, debugLoc, IR->scopebb());
}

// Sets the (current) value for a debuginfo variable.
void ldc::DIBuilder::SetValue(const Loc &loc, llvm::Value *value,
                              ldc::DILocalVariable divar,
                              ldc::DIExpression diexpr) {
  unsigned charnum = (loc.linnum ? loc.charnum : 0);
  auto debugLoc = llvm::DebugLoc::get(loc.linnum, charnum, GetCurrentScope());
  DBuilder.insertDbgValueIntrinsic(value,
#if LDC_LLVM_VER < 600
                                   0,
#endif
                                   divar, diexpr, debugLoc, IR->scopebb());
}

ldc::DIFile ldc::DIBuilder::CreateFile(Loc &loc) {
  const char* filename = loc.filename;
  if (!filename)
    filename = IR->dmodule->srcfile->toChars();
  llvm::SmallString<128> path(filename);
  llvm::sys::fs::make_absolute(path);

  return DBuilder.createFile(llvm::sys::path::filename(path),
                             llvm::sys::path::parent_path(path));
}

ldc::DIFile ldc::DIBuilder::CreateFile() {
  Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
  return CreateFile(loc);
}

ldc::DIFile ldc::DIBuilder::CreateFile(Dsymbol* decl) {
  Loc loc;
  for (Dsymbol* sym = decl; sym && !loc.filename; sym = sym->parent)
    loc = sym->loc;
  return loc.filename ? CreateFile(loc) : CreateFile();
}

ldc::DIType ldc::DIBuilder::CreateBasicType(Type *type) {
  using namespace llvm::dwarf;

  Type *t = type->toBasetype();
  llvm::Type *T = DtoType(type);

  // find encoding
  unsigned Encoding;
  switch (t->ty) {
  case Tbool:
    Encoding = DW_ATE_boolean;
    break;
  case Tchar:
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // VS debugger does not support DW_ATE_UTF for char
      Encoding = DW_ATE_unsigned_char;
      break;
    }
    // fall through
  case Twchar:
  case Tdchar:
    Encoding = DW_ATE_UTF;
    break;
  case Tint8:
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // VS debugger does not support DW_ATE_signed for 8-bit
      Encoding = DW_ATE_signed_char;
      break;
    }
    // fall through
  case Tint16:
  case Tint32:
  case Tint64:
  case Tint128:
    Encoding = DW_ATE_signed;
    break;
  case Tuns8:
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // VS debugger does not support DW_ATE_unsigned for 8-bit
      Encoding = DW_ATE_unsigned_char;
      break;
    }
    // fall through
  case Tuns16:
  case Tuns32:
  case Tuns64:
  case Tuns128:
    Encoding = DW_ATE_unsigned;
    break;
  case Tfloat32:
  case Tfloat64:
  case Tfloat80:
    Encoding = DW_ATE_float;
    break;
  case Timaginary32:
  case Timaginary64:
  case Timaginary80:
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // DW_ATE_imaginary_float not supported by the LLVM DWARF->CodeView
      // conversion
      Encoding = DW_ATE_float;
      break;
    }
    Encoding = DW_ATE_imaginary_float;
    break;
  case Tcomplex32:
  case Tcomplex64:
  case Tcomplex80:
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // DW_ATE_complex_float not supported by the LLVM DWARF->CodeView
      // conversion
      return CreateComplexType(t);
    }
    Encoding = DW_ATE_complex_float;
    break;
  default:
    llvm_unreachable(
        "Unsupported basic type for debug info in DIBuilder::CreateBasicType");
  }

  return DBuilder.createBasicType(type->toChars(),         // name
                                  getTypeAllocSize(T) * 8, // size (bits)
#if LDC_LLVM_VER < 400
                                  getABITypeAlign(T) * 8,  // align (bits)
#endif
                                  Encoding);
}

ldc::DIType ldc::DIBuilder::CreateEnumType(Type *type) {
  assert(type->ty == Tenum);

  llvm::Type *T = DtoType(type);
  TypeEnum *te = static_cast<TypeEnum *>(type);

  if (te->sym->isSpecial()) {
    return CreateBasicType(te->sym->memtype);
  }

  llvm::SmallVector<LLMetadata *, 8> subscripts;
  for (auto m : *te->sym->members) {
    EnumMember *em = m->isEnumMember();
    llvm::StringRef Name(em->toChars());
    uint64_t Val = em->value()->toInteger();
    auto Subscript = DBuilder.createEnumerator(Name, Val);
    subscripts.push_back(Subscript);
  }

  llvm::StringRef Name = te->sym->toPrettyChars(true);
  unsigned LineNumber = te->sym->loc.linnum;
  ldc::DIFile File(CreateFile(te->sym));

  return DBuilder.createEnumerationType(
      GetCU(), Name, File, LineNumber,
      getTypeAllocSize(T) * 8,               // size (bits)
      getABITypeAlign(T) * 8,                // align (bits)
      DBuilder.getOrCreateArray(subscripts), // subscripts
      CreateTypeDescription(te->sym->memtype));
}

ldc::DIType ldc::DIBuilder::CreatePointerType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();
  assert(t->ty == Tpointer);

  // find base type
  Type *nt = t->nextOf();
  // translate void pointers to byte pointers
  if (nt->toBasetype()->ty == Tvoid)
    nt = Type::tuns8;

#if LDC_LLVM_VER >= 500
  // TODO: The addressspace is important for dcompute targets.
  // See e.g. https://www.mail-archive.com/dwarf-discuss@lists.dwarfstd.org/msg00326.html
  const llvm::Optional<unsigned> DWARFAddressSpace = llvm::None;
#endif

  return DBuilder.createPointerType(CreateTypeDescription(nt),
                                    getTypeAllocSize(T) * 8, // size (bits)
                                    getABITypeAlign(T) * 8,  // align (bits)
#if LDC_LLVM_VER >= 500
                                    DWARFAddressSpace,
#endif
                                    type->toPrettyChars(true) // name
                                    );
}

ldc::DIType ldc::DIBuilder::CreateVectorType(Type *type) {
  LLType *T = DtoType(type);
  Type *t = type->toBasetype();

  assert(t->ty == Tvector &&
         "Only vectors allowed for debug info in DIBuilder::CreateVectorType");
  TypeVector *tv = static_cast<TypeVector *>(t);
  Type *te = tv->elementType();
  // translate void vectors to byte vectors
  if (te->toBasetype()->ty == Tvoid)
    te = Type::tuns8;
  int64_t Dim = tv->size(Loc()) / te->size(Loc());
  LLMetadata *subscripts[] = {DBuilder.getOrCreateSubrange(0, Dim)};

  return DBuilder.createVectorType(
      getTypeAllocSize(T) * 8,              // size (bits)
      getABITypeAlign(T) * 8,               // align (bits)
      CreateTypeDescription(te),            // element type
      DBuilder.getOrCreateArray(subscripts) // subscripts
      );
}

ldc::DIType ldc::DIBuilder::CreateComplexType(Type *type) {
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    Type* elemtype = nullptr;
    switch (t->ty) {
    case Tcomplex32:
        elemtype = Type::tfloat32;
        break;
    case Tcomplex64:
        elemtype = Type::tfloat64;
        break;
    case Tcomplex80:
        elemtype = Type::tfloat80;
        break;
    default:
        llvm_unreachable(
            "Unexpected type for debug info in DIBuilder::CreateComplexType");
    }
    ldc::DIFile file = CreateFile();

    auto imoffset = getTypeAllocSize(DtoType(elemtype));
    LLMetadata *elems[] = {
        CreateMemberType(0, elemtype, file, "re", 0, Prot::public_),
        CreateMemberType(0, elemtype, file, "im", imoffset, Prot::public_)};

    return DBuilder.createStructType(GetCU(),
                                     t->toChars(),            // Name
                                     file,                    // File
                                     0,                       // LineNo
                                     getTypeAllocSize(T) * 8, // size in bits
                                     getABITypeAlign(T) * 8,  // alignment
                                     DIFlagZero,              // What here?
                                     getNullDIType(),         // derived from
                                     DBuilder.getOrCreateArray(elems),
                                     0,               // RunTimeLang
                                     getNullDIType(), // VTableHolder
                                     uniqueIdent(t)); // UniqueIdentifier
}

ldc::DIType ldc::DIBuilder::CreateMemberType(unsigned linnum, Type *type,
                                             ldc::DIFile file,
                                             const char *c_name,
                                             unsigned offset, Prot::Kind prot) {
  Type *t = type->toBasetype();

  // translate functions to function pointers
  if (t->ty == Tfunction)
    t = t->pointerTo();

  llvm::Type *T = DtoType(t);

  // find base type
  ldc::DIType basetype = CreateTypeDescription(t);

  auto Flags = DIFlagZero;
  switch (prot) {
  case Prot::private_:
    Flags = DIFlags::FlagPrivate;
    break;
  case Prot::protected_:
    Flags = DIFlags::FlagProtected;
    break;
  case Prot::public_:
    Flags = DIFlags::FlagPublic;
    break;
  default:
    break;
  }

  return DBuilder.createMemberType(GetCU(),
                                   c_name,                  // name
                                   file,                    // file
                                   linnum,                  // line number
                                   getTypeAllocSize(T) * 8, // size (bits)
                                   getABITypeAlign(T) * 8,  // align (bits)
                                   offset * 8,              // offset (bits)
                                   Flags,                   // flags
                                   basetype                 // derived from
                                   );
}

void ldc::DIBuilder::AddFields(AggregateDeclaration *ad, ldc::DIFile file,
                               llvm::SmallVector<LLMetadata *, 16> &elems) {
  size_t narr = ad->fields.dim;
  elems.reserve(narr);
  for (auto vd : ad->fields) {
    elems.push_back(CreateMemberType(vd->loc.linnum, vd->type, file,
                                     vd->toChars(), vd->offset,
                                     vd->prot().kind));
  }
}

ldc::DIType ldc::DIBuilder::CreateCompositeType(Type *type) {
  Type *t = type->toBasetype();
  assert((t->ty == Tstruct || t->ty == Tclass) &&
         "Unsupported type for debug info in DIBuilder::CreateCompositeType");
  AggregateDeclaration *ad;
  if (t->ty == Tstruct) {
    TypeStruct *ts = static_cast<TypeStruct *>(t);
    ad = ts->sym;
  } else {
    TypeClass *tc = static_cast<TypeClass *>(t);
    ad = tc->sym;
  }
  assert(ad);

  // Use the actual type associated with the declaration, ignoring any
  // const/wrappers.
  LLType *T = DtoType(ad->type);
  if (t->ty == Tclass)
    T = T->getPointerElementType();
  IrTypeAggr *ir = ad->type->ctype->isAggr();
  assert(ir);

  if (static_cast<llvm::MDNode *>(ir->diCompositeType) != nullptr) {
    return ir->diCompositeType;
  }

  const llvm::StringRef name =
      (ad->isClassDeclaration() && ad->isClassDeclaration()->isCPPinterface()
           ? ad->ident->toChars()
           : ad->toPrettyChars(true));

  // if we don't know the aggregate's size, we don't know enough about it
  // to provide debug info. probably a forward-declared struct?
  if (ad->sizeok == SIZEOKnone) {
    return DBuilder.createUnspecifiedType(name);
  }

  // elements
  llvm::SmallVector<LLMetadata *, 16> elems;

  // defaults
  unsigned linnum = ad->loc.linnum;
  ldc::DICompileUnit CU(GetCU());
  assert(CU && "Compilation unit missing or corrupted");
  ldc::DIFile file = CreateFile(ad);
  ldc::DIType derivedFrom = getNullDIType();

  // set diCompositeType to handle recursive types properly
  unsigned tag = (t->ty == Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                    : llvm::dwarf::DW_TAG_class_type;
  ir->diCompositeType = DBuilder.createReplaceableCompositeType(
      tag, name, CU, file, linnum);

  if (!ad->isInterfaceDeclaration()) // plain interfaces don't have one
  {
    ClassDeclaration *classDecl = ad->isClassDeclaration();
    if (classDecl && classDecl->baseClass) {
      derivedFrom = CreateCompositeType(classDecl->baseClass->getType());
      // needs a forward declaration to add inheritence information to elems
      ldc::DIType fwd =
          DBuilder.createClassType(CU,     // compile unit where defined
                                   name,   // name
                                   file,   // file where defined
                                   linnum, // line number where defined
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   0,                       // offset in bits,
                                   DIFlags::FlagFwdDecl,    // flags
                                   derivedFrom,             // DerivedFrom
                                   getEmptyDINodeArray(),
                                   getNullDIType(), // VTableHolder
                                   nullptr,         // TemplateParms
                                   uniqueIdent(t)); // UniqueIdentifier
      auto dt = DBuilder.createInheritance(fwd, derivedFrom, 0,
                                           DIFlags::FlagPublic);
      elems.push_back(dt);
    }
    AddFields(ad, file, elems);
  }

  auto elemsArray = DBuilder.getOrCreateArray(elems);

  ldc::DIType ret;
  if (t->ty == Tclass) {
    ret = DBuilder.createClassType(CU,     // compile unit where defined
                                   name,   // name
                                   file,   // file where defined
                                   linnum, // line number where defined
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   0,                       // offset in bits,
                                   DIFlagZero,              // flags
                                   derivedFrom,             // DerivedFrom
                                   elemsArray,
                                   getNullDIType(), // VTableHolder
                                   nullptr,         // TemplateParms
                                   uniqueIdent(t)); // UniqueIdentifier
  } else {
    ret = DBuilder.createStructType(CU,     // compile unit where defined
                                    name,   // name
                                    file,   // file where defined
                                    linnum, // line number where defined
                                    getTypeAllocSize(T) * 8, // size in bits
                                    getABITypeAlign(T) * 8, // alignment in bits
                                    DIFlagZero,             // flags
                                    derivedFrom,            // DerivedFrom
                                    elemsArray,
                                    0,               // RunTimeLang
                                    getNullDIType(), // VTableHolder
                                    uniqueIdent(t)); // UniqueIdentifier
  }

  ir->diCompositeType = DBuilder.replaceTemporary(
      llvm::TempDINode(ir->diCompositeType), static_cast<llvm::DIType *>(ret));
  ir->diCompositeType = ret;

  return ret;
}

ldc::DIType ldc::DIBuilder::CreateArrayType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();
  assert(t->ty == Tarray);

  ldc::DIFile file = CreateFile();

  LLMetadata *elems[] = {
      CreateMemberType(0, Type::tsize_t, file, "length", 0, Prot::public_),
      CreateMemberType(0, t->nextOf()->pointerTo(), file, "ptr",
                       global.params.is64bit ? 8 : 4, Prot::public_)};

  return DBuilder.createStructType(GetCU(),
                                   type->toPrettyChars(true), // Name
                                   file,                    // File
                                   0,                       // LineNo
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   DIFlagZero,              // What here?
                                   getNullDIType(),         // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,               // RunTimeLang
                                   getNullDIType(), // VTableHolder
                                   uniqueIdent(t)); // UniqueIdentifier
}

ldc::DIType ldc::DIBuilder::CreateSArrayType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();
  assert(t->ty == Tsarray);

  // find base type
  llvm::SmallVector<LLMetadata *, 8> subscripts;
  while (t->ty == Tsarray) {
    TypeSArray *tsa = static_cast<TypeSArray *>(t);
    int64_t Count = tsa->dim->toInteger();
    auto subscript = DBuilder.getOrCreateSubrange(0, Count);
    subscripts.push_back(subscript);
    t = t->nextOf();
  }

  // element type: void => byte, function => function pointer
  t = t->toBasetype();
  if (t->ty == Tvoid)
    t = Type::tuns8;
  else if (t->ty == Tfunction)
    t = t->pointerTo();

  return DBuilder.createArrayType(
      getTypeAllocSize(T) * 8,              // size (bits)
      getABITypeAlign(T) * 8,               // align (bits)
      CreateTypeDescription(t),             // element type
      DBuilder.getOrCreateArray(subscripts) // subscripts
      );
}

ldc::DIType ldc::DIBuilder::CreateAArrayType(Type *type) {
  return CreatePointerType(Type::tvoidptr);
}

////////////////////////////////////////////////////////////////////////////////

ldc::DISubroutineType ldc::DIBuilder::CreateFunctionType(Type *type) {
  assert(type->toBasetype()->ty == Tfunction);

  TypeFunction *t = static_cast<TypeFunction *>(type);
  Type *retType = t->next;

  // Create "dummy" subroutine type for the return type
  LLMetadata *params = {CreateTypeDescription(retType)};
  auto paramsArray = DBuilder.getOrCreateTypeArray(params);

#if LDC_LLVM_VER >= 308
  return DBuilder.createSubroutineType(paramsArray);
#else
  return DBuilder.createSubroutineType(CreateFile(), paramsArray);
#endif
}

ldc::DISubroutineType ldc::DIBuilder::CreateEmptyFunctionType() {
  auto paramsArray = DBuilder.getOrCreateTypeArray(llvm::None);
#if LDC_LLVM_VER >= 308
  return DBuilder.createSubroutineType(paramsArray);
#else
  return DBuilder.createSubroutineType(CreateFile(), paramsArray);
#endif
}

ldc::DIType ldc::DIBuilder::CreateDelegateType(Type *type) {
  assert(type->toBasetype()->ty == Tdelegate);

  llvm::Type *T = DtoType(type);
  auto t = static_cast<TypeDelegate *>(type);

  ldc::DICompileUnit CU(GetCU());
  assert(CU && "Compilation unit missing or corrupted");
  auto file = CreateFile();

  LLMetadata *elems[] = {
      CreateMemberType(0, Type::tvoidptr, file, "context", 0, Prot::public_),
      CreateMemberType(0, t->next, file, "funcptr",
                       global.params.is64bit ? 8 : 4, Prot::public_)};

  return DBuilder.createStructType(CU,           // compile unit where defined
                                   type->toPrettyChars(true), // name
                                   file,         // file where defined
                                   0,            // line number where defined
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   DIFlagZero,              // flags
                                   getNullDIType(),         // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,               // RunTimeLang
                                   getNullDIType(), // VTableHolder
                                   uniqueIdent(t)); // UniqueIdentifier
}

////////////////////////////////////////////////////////////////////////////////
bool isOpaqueEnumType(Type *type) {
  if (type->ty != Tenum)
    return false;

  TypeEnum *te = static_cast<TypeEnum *>(type);
  return !te->sym->memtype;
}

ldc::DIType ldc::DIBuilder::CreateTypeDescription(Type *type) {
  // Check for opaque enum first, Bugzilla 13792
  if (isOpaqueEnumType(type)) {
    const auto ed = static_cast<TypeEnum *>(type)->sym;
    return DBuilder.createUnspecifiedType(ed->toPrettyChars(true));
  }

  Type *t = type->toBasetype();

  if (t->ty == Tvoid)
#if LDC_LLVM_VER >= 309
    return nullptr;
#else
    return DBuilder.createUnspecifiedType(type->toPrettyChars(true));
#endif
  if (t->ty == Tnull) // display null as void*
    return DBuilder.createPointerType(CreateTypeDescription(Type::tvoid),
                                      8, 8,
#if LDC_LLVM_VER >= 500
                                      /* DWARFAddressSpace */ llvm::None,
#endif

                                      "typeof(null)");
  if (t->ty == Tvector)
    return CreateVectorType(type);
  if (t->isintegral() || t->isfloating()) {
    if (type->ty == Tenum)
      return CreateEnumType(type);
    return CreateBasicType(type);
  }
  if (t->ty == Tpointer)
    return CreatePointerType(type);
  if (t->ty == Tarray)
    return CreateArrayType(type);
  if (t->ty == Tsarray)
    return CreateSArrayType(type);
  if (t->ty == Taarray)
    return CreateAArrayType(type);
  if (t->ty == Tstruct)
    return CreateCompositeType(type);
  if (t->ty == Tclass) {
    LLType* T = DtoType(t);
    const auto aggregateDIType = CreateCompositeType(type);
    const auto name = (aggregateDIType->getName() + "*").str();
    return DBuilder.createPointerType(aggregateDIType,
                                      getTypeAllocSize(T) * 8, getABITypeAlign(T) * 8,
#if LDC_LLVM_VER >= 500
                                      llvm::None,
#endif
                                      name);
  }
  if (t->ty == Tfunction)
    return CreateFunctionType(type);
  if (t->ty == Tdelegate)
    return CreateDelegateType(type);

  // Crash if the type is not supported.
  llvm_unreachable("Unsupported type in debug info");
}

////////////////////////////////////////////////////////////////////////////////

#if LDC_LLVM_VER >= 309
using DebugEmissionKind = llvm::DICompileUnit::DebugEmissionKind;
#else
using DebugEmissionKind = llvm::DIBuilder::DebugEmissionKind;
#endif

DebugEmissionKind getDebugEmissionKind()
{
#if LDC_LLVM_VER >= 309
  switch (global.params.symdebug)
  {
    case 0:
      return llvm::DICompileUnit::NoDebug;
    case 1:
    case 2:
      return llvm::DICompileUnit::FullDebug;
    case 3:
      return llvm::DICompileUnit::LineTablesOnly;
    default:
      llvm_unreachable("unknown DebugEmissionKind");
  }
#else
  assert(global.params.symdebug != 0);
  return global.params.symdebug == 3 ?
      llvm::DIBuilder::LineTablesOnly :
      llvm::DIBuilder::FullDebug;
#endif
}



void ldc::DIBuilder::EmitCompileUnit(Module *m) {
  if (!mustEmitLocationsDebugInfo()) {
    return;
  }

  Logger::println("D to dwarf compile_unit");
  LOG_SCOPE;

  assert(!CUNode && "Already created compile unit for this DIBuilder instance");

  // prepare srcpath
  llvm::SmallString<128> srcpath(m->srcfile->name->toChars());
  llvm::sys::fs::make_absolute(srcpath);

  // prepare producer name string
  auto producerName = std::string("LDC ") + ldc::ldc_version + " (LLVM " +
                      ldc::llvm_version + ")";

#if LDC_LLVM_VER >= 308
  if (global.params.targetTriple->isWindowsMSVCEnvironment())
    IR->module.addModuleFlag(llvm::Module::Warning, "CodeView", 1);
  else if (global.params.dwarfVersion > 0)
    IR->module.addModuleFlag(llvm::Module::Warning, "Dwarf Version",
                             global.params.dwarfVersion);
#endif
  // Metadata without a correct version will be stripped by UpgradeDebugInfo.
  IR->module.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);

  CUNode = DBuilder.createCompileUnit(
      global.params.symdebug == 2 ? llvm::dwarf::DW_LANG_C
                                  : llvm::dwarf::DW_LANG_D,
#if LDC_LLVM_VER >= 400
      DBuilder.createFile(llvm::sys::path::filename(srcpath),
                          llvm::sys::path::parent_path(srcpath)),
#else
      llvm::sys::path::filename(srcpath), llvm::sys::path::parent_path(srcpath),
#endif
      producerName,
      isOptimizationEnabled(), // isOptimized
      llvm::StringRef(),       // Flags TODO
      1,                       // Runtime Version TODO
      llvm::StringRef(),       // SplitName
      getDebugEmissionKind(),  // DebugEmissionKind
      0                        // DWOId
#if LDC_LLVM_VER < 309
      , mustEmitFullDebugInfo()  // EmitDebugInfo
#endif
  );
}

ldc::DISubprogram ldc::DIBuilder::EmitSubProgram(FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

  ldc::DIFile file = CreateFile(fd);

  // Create subroutine type
  ldc::DISubroutineType DIFnType = mustEmitFullDebugInfo() ?
      CreateFunctionType(static_cast<TypeFunction *>(fd->type)) :
      CreateEmptyFunctionType();

  // FIXME: duplicates?
  auto SP = DBuilder.createFunction(
      CU,                                 // context
      fd->toPrettyChars(true),            // name
      getIrFunc(fd)->getLLVMFuncName(),   // linkage name
      file,                               // file
      fd->loc.linnum,                     // line no
      DIFnType,                           // type
      fd->protection.kind == Prot::private_, // is local to unit
      true,                               // isdefinition
      fd->loc.linnum,                     // FIXME: scope line
      DIFlags::FlagPrototyped,            // Flags
      isOptimizationEnabled()             // isOptimized
#if LDC_LLVM_VER < 308
      ,
      DtoFunction(fd)
#endif
      );
#if LDC_LLVM_VER >= 308
  DtoFunction(fd)->setSubprogram(SP);
#endif
  return SP;
}

ldc::DISubprogram ldc::DIBuilder::EmitThunk(llvm::Function *Thunk,
                                            FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  Logger::println("Thunk to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU && "Compilation unit missing or corrupted in DIBuilder::EmitThunk");

  ldc::DIFile file = CreateFile(fd);

  // Create subroutine type (thunk has same type as wrapped function)
  ldc::DISubroutineType DIFnType = CreateFunctionType(fd->type);

  std::string name = fd->toPrettyChars(true);
  name.append(".__thunk");

  // FIXME: duplicates?
  auto SP = DBuilder.createFunction(
      CU,                                 // context
      name,                               // name
      Thunk->getName(),                   // linkage name
      file,                               // file
      fd->loc.linnum,                     // line no
      DIFnType,                           // type
      fd->protection.kind == Prot::private_, // is local to unit
      true,                               // isdefinition
      fd->loc.linnum,                     // FIXME: scope line
      DIFlags::FlagPrototyped,            // Flags
      isOptimizationEnabled()             // isOptimized
#if LDC_LLVM_VER < 308
      ,
      DtoFunction(fd)
#endif
      );
#if LDC_LLVM_VER >= 308
  if (fd->fbody)
    DtoFunction(fd)->setSubprogram(SP);
#endif
  return SP;
}

ldc::DISubprogram ldc::DIBuilder::EmitModuleCTor(llvm::Function *Fn,
                                                 llvm::StringRef prettyname) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");
  ldc::DIFile file = CreateFile();

  // Create "dummy" subroutine type for the return type
  LLMetadata *params = {CreateTypeDescription(Type::tvoid)};
  auto paramsArray = DBuilder.getOrCreateTypeArray(params);
#if LDC_LLVM_VER >= 308
  auto DIFnType = DBuilder.createSubroutineType(paramsArray);
#else
  auto DIFnType = DBuilder.createSubroutineType(file, paramsArray);
#endif

  // FIXME: duplicates?
  auto SP =
      DBuilder.createFunction(CU,            // context
                              prettyname,    // name
                              Fn->getName(), // linkage name
                              file,          // file
                              0,             // line no
                              DIFnType,      // return type. TODO: fill it up
                              true,          // is local to unit
                              true,          // isdefinition
                              0,             // FIXME: scope line
                              DIFlags::FlagPrototyped | DIFlags::FlagArtificial,
                              isOptimizationEnabled() // isOptimized
#if LDC_LLVM_VER < 308
                              ,
                              Fn
#endif
                              );
#if LDC_LLVM_VER >= 308
  Fn->setSubprogram(SP);
#endif
  return SP;
}

void ldc::DIBuilder::EmitFuncStart(FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf funcstart");
  LOG_SCOPE;

  assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
  EmitStopPoint(fd->loc);
}

void ldc::DIBuilder::EmitFuncEnd(FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf funcend");
  LOG_SCOPE;

  assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
  EmitStopPoint(fd->endloc);
}

void ldc::DIBuilder::EmitBlockStart(Loc &loc) {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf block start");
  LOG_SCOPE;

  ldc::DILexicalBlock block =
      DBuilder.createLexicalBlock(GetCurrentScope(),           // scope
                                  CreateFile(loc),             // file
                                  loc.linnum,                  // line
                                  loc.linnum ? loc.charnum : 0 // column
                                  );
  IR->func()->diLexicalBlocks.push(block);
  EmitStopPoint(loc);
}

void ldc::DIBuilder::EmitBlockEnd() {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf block end");
  LOG_SCOPE;

  IrFunction *fn = IR->func();
  assert(!fn->diLexicalBlocks.empty());
  fn->diLexicalBlocks.pop();
}

void ldc::DIBuilder::EmitStopPoint(Loc &loc) {
  if (!mustEmitLocationsDebugInfo())
    return;

  // If we already have a location set and the current loc is invalid
  // (line 0), then we can just ignore it (see GitHub issue #998 for why we
  // cannot do this in all cases).
  if (!loc.linnum && IR->ir->getCurrentDebugLocation())
    return;
  unsigned linnum = loc.linnum;
  // without proper loc use the line of the enclosing symbol that has line
  // number debug info
  for (Dsymbol *sym = IR->func()->decl; sym && !linnum; sym = sym->parent)
    linnum = sym->loc.linnum;
  if (!linnum)
    linnum = 1;

  unsigned charnum = (loc.linnum ? loc.charnum : 0);
  Logger::println("D to dwarf stoppoint at line %u, column %u", linnum,
                  charnum);
  LOG_SCOPE;
  IR->ir->SetCurrentDebugLocation(
      llvm::DebugLoc::get(linnum, charnum, GetCurrentScope()));
  currentLoc = loc;
}

Loc ldc::DIBuilder::GetCurrentLoc() const { return currentLoc; }

void ldc::DIBuilder::EmitValue(llvm::Value *val, VarDeclaration *vd) {
  auto sub = IR->func()->variableMap.find(vd);
  if (sub == IR->func()->variableMap.end())
    return;

  ldc::DILocalVariable debugVariable = sub->second;
  if (!mustEmitFullDebugInfo() || !debugVariable)
    return;

  llvm::Instruction *instr =
      DBuilder.insertDbgValueIntrinsic(val,
#if LDC_LLVM_VER < 600
                                       0, 
#endif
                                       debugVariable,
                                       DBuilder.createExpression(),
                                       IR->ir->getCurrentDebugLocation(),
                                       IR->scopebb());
  instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

void ldc::DIBuilder::EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                                       Type *type, bool isThisPtr,
                                       bool forceAsLocal, bool isRefRVal,
                                       llvm::ArrayRef<int64_t> addr) {
  if (!mustEmitFullDebugInfo())
    return;

  Logger::println("D to dwarf local variable");
  LOG_SCOPE;

  auto &variableMap = IR->func()->variableMap;
  auto sub = variableMap.find(vd);
  if (sub != variableMap.end())
    return; // ensure that the debug variable is created only once

  // get type description
  if (!type)
    type = vd->type;
  ldc::DIType TD = CreateTypeDescription(type);
  if (static_cast<llvm::MDNode *>(TD) == nullptr)
    return; // unsupported

  const bool isRefOrOut = vd->isRef() || vd->isOut(); // incl. special-ref vars

  // For MSVC x64 targets, declare params rewritten by ExplicitByvalRewrite as
  // DI references, as if they were ref parameters.
  const bool isPassedExplicitlyByval =
      isTargetMSVCx64 && !isRefOrOut && isaArgument(ll) && addr.empty();

  bool useDbgValueIntrinsic = false;
  if (isRefOrOut || isPassedExplicitlyByval) {
    // With the exception of special-ref loop variables, the reference/pointer
    // itself is constant. So we don't have to attach the debug information to a
    // memory location and can use llvm.dbg.value to set the constant pointer
    // for the DI reference.
    useDbgValueIntrinsic =
        isPassedExplicitlyByval || (!isSpecialRefVar(vd) && isRefRVal);
#if LDC_LLVM_VER >= 308
    // Note: createReferenceType expects the size to be the size of a pointer,
    // not the size of the type the reference refers to.
    TD = DBuilder.createReferenceType(
        llvm::dwarf::DW_TAG_reference_type, TD,
        gDataLayout->getPointerSizeInBits(), // size (bits)
        DtoAlignment(type) * 8);             // align (bits)
#else
    TD = DBuilder.createReferenceType(llvm::dwarf::DW_TAG_reference_type, TD);
#endif
  } else {
    // FIXME: For MSVC x64 targets, declare dynamic array and vector parameters
    //        as DI locals to work around garbage for both cdb and VS debuggers.
    if (isTargetMSVCx64) {
      TY ty = type->toBasetype()->ty;
      if (ty == Tarray || ty == Tvector)
        forceAsLocal = true;
    }
  }

  // get variable description
  assert(!vd->isDataseg() && "static variable");

#if LDC_LLVM_VER < 308
  unsigned tag;
  if (!forceAsLocal && vd->isParameter()) {
    tag = llvm::dwarf::DW_TAG_arg_variable;
  } else {
    tag = llvm::dwarf::DW_TAG_auto_variable;
  }
#endif

  ldc::DILocalVariable debugVariable;
  auto Flags = !isThisPtr
                   ? DIFlagZero
                   : DIFlags::FlagArtificial | DIFlags::FlagObjectPointer;

#if LDC_LLVM_VER < 308
  debugVariable = DBuilder.createLocalVariable(tag,                 // tag
                                               GetCurrentScope(),   // scope
                                               vd->toChars(),       // name
                                               CreateFile(vd),      // file
                                               vd->loc.linnum,      // line num
                                               TD,                  // type
                                               true,                // preserve
                                               Flags                // flags
                                               );
#else
  if (!forceAsLocal && vd->isParameter()) {
    FuncDeclaration *fd = vd->parent->isFuncDeclaration();
    assert(fd);
    size_t argNo = 0;
    if (fd->vthis != vd) {
      assert(fd->parameters);
      auto it = std::find(fd->parameters->begin(), fd->parameters->end(), vd);
      assert(it != fd->parameters->end());
      argNo = it - fd->parameters->begin();
      if (fd->vthis)
        argNo++;
    }

    debugVariable = DBuilder.createParameterVariable(GetCurrentScope(), // scope
                                                     vd->toChars(),     // name
                                                     argNo + 1,
                                                     CreateFile(vd), // file
                                                     vd->loc.linnum, // line num
                                                     TD,             // type
                                                     true,           // preserve
                                                     Flags           // flags
                                                     );
  } else {
    debugVariable = DBuilder.createAutoVariable(GetCurrentScope(), // scope
                                                vd->toChars(),     // name
                                                CreateFile(vd),    // file
                                                vd->loc.linnum,    // line num
                                                TD,                // type
                                                true,              // preserve
                                                Flags              // flags
                                                );
  }
#endif
  variableMap[vd] = debugVariable;

  if (useDbgValueIntrinsic) {
    SetValue(vd->loc, ll, debugVariable,
             addr.empty() ? DBuilder.createExpression()
                          : DBuilder.createExpression(addr));
  } else {
    Declare(vd->loc, ll, debugVariable,
            addr.empty() ? DBuilder.createExpression()
                         : DBuilder.createExpression(addr));
  }
}

void ldc::DIBuilder::EmitGlobalVariable(llvm::GlobalVariable *llVar,
                                        VarDeclaration *vd) {
  if (!mustEmitFullDebugInfo())
    return;

  Logger::println("D to dwarf global_variable");
  LOG_SCOPE;

  assert(vd->isDataseg() ||
         (vd->storage_class & (STCconst | STCimmutable) && vd->_init));

  OutBuffer mangleBuf;
  mangleToBuffer(vd, &mangleBuf);

#if LDC_LLVM_VER >= 400
  auto DIVar = DBuilder.createGlobalVariableExpression(
#else
  DBuilder.createGlobalVariable(
#endif
      GetCU(),                                // context
      vd->toChars(),                          // name
      mangleBuf.peekString(),                 // linkage name
      CreateFile(vd),                         // file
      vd->loc.linnum,                         // line num
      CreateTypeDescription(vd->type),        // type
      vd->protection.kind == Prot::private_,     // is local to unit
#if LDC_LLVM_VER >= 400
      nullptr // relative location of field
#else
      llVar // value
#endif
      );

#if LDC_LLVM_VER >= 400
  llVar->addDebugInfo(DIVar);
#endif
}

void ldc::DIBuilder::Finalize() {
  if (!mustEmitLocationsDebugInfo())
    return;

  DBuilder.finalize();
}
