//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dibuilder.h"

#include "dmd/declaration.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/import.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/nspace.h"
#include "dmd/root/dcompat.h"
#include "dmd/target.h"
#include "dmd/template.h"
#include "driver/cl_options.h"
#include "driver/ldc-version.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"
#include "ir/irmodule.h"
#include "ir/irtypeaggr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <functional>

////////////////////////////////////////////////////////////////////////////////

namespace cl = llvm::cl;
using LLMetadata = llvm::Metadata;

#if LDC_LLVM_VER >= 1600
namespace llvm {
  template <typename T> using Optional = std::optional<T>;
  inline constexpr std::nullopt_t None = std::nullopt;
}
#endif

static cl::opt<cl::boolOrDefault> emitColumnInfo(
    "gcolumn-info", cl::ZeroOrMore, cl::Hidden,
    cl::desc("Include column numbers in line debug infos. Defaults to "
             "true for non-MSVC targets."));

namespace ldc {

// in gen/cpp-imitating-naming.d
const char *convertDIdentifierToCPlusPlus(const char *name,
                                          d_size_t nameLength);

namespace {
llvm::StringRef uniqueIdent(Type *t) {
  if (t->deco)
    return t->deco;
  return llvm::StringRef();
}

const char *getTemplateInstanceName(TemplateInstance *ti) {
  const auto realParent = ti->parent;
  ti->parent = nullptr;
  const auto name = ti->toPrettyChars(true);
  ti->parent = realParent;
  return name;
}

llvm::StringRef processDIName(llvm::StringRef name) {
  return global.params.symdebug == 2
             ? convertDIdentifierToCPlusPlus(name.data(), name.size())
             : name;
}

} // namespace

bool DIBuilder::mustEmitFullDebugInfo() {
  // only for -g and -gc
  // TODO: but not dcompute (yet)

  if (IR->dcomputetarget)
    return false;

  return global.params.symdebug == 1 || global.params.symdebug == 2;
}

bool DIBuilder::mustEmitLocationsDebugInfo() {
  // for -g -gc and -gline-tables-only
  // TODO:but not dcompute (yet)

  if (IR->dcomputetarget)
    return false;

  return (global.params.symdebug > 0) || global.params.outputSourceLocations;
}

////////////////////////////////////////////////////////////////////////////////

DIBuilder::DIBuilder(IRState *const IR)
    : IR(IR), DBuilder(IR->module), CUNode(nullptr),
      emitCodeView(!opts::emitDwarfDebugInfo &&
                   global.params.targetTriple->isWindowsMSVCEnvironment()),
      // like clang, don't emit any column infos for CodeView by default
      // (https://reviews.llvm.org/D23720)
      emitColumnInfo(opts::getFlagOrDefault(::emitColumnInfo, !emitCodeView)) {}

unsigned DIBuilder::getColumn(const Loc &loc) const {
  return (loc.linnum() && emitColumnInfo) ? loc.charnum() : 0;
}

// Returns the DI scope of a symbol.
DIScope DIBuilder::GetSymbolScope(Dsymbol *s) {
  // don't recreate parent entries if we only need location debug info
  if (!mustEmitFullDebugInfo())
    return GetCU();

  auto parent = s->toParent();

  auto vd = s->isVarDeclaration();
  if (vd && vd->isDataseg()) {
    // static variables get attached to the module scope, but their
    // parent composite types have to get declared
    while (!parent->isModule()) {
      if (parent->isAggregateDeclaration())
        CreateCompositeType(parent->getType());
      parent = parent->toParent();
    }
  }

  if (auto ti = parent->isTemplateInstance()) {
    return EmitNamespace(ti, getTemplateInstanceName(ti));
  } else if (auto m = parent->isModule()) {
    return EmitModule(m);
  } else if (parent->isAggregateDeclaration()) {
    return CreateCompositeType(parent->getType());
  } else if (auto fd = parent->isFuncDeclaration()) {
    DtoDeclareFunction(fd);
    return EmitSubProgram(fd);
  } else if (auto ns = parent->isNspace()) {
    return EmitNamespace(ns, ns->toChars());
  } else if (auto fwd = parent->isForwardingScopeDsymbol()) {
    return GetSymbolScope(fwd);
  } else if (auto ed = parent->isEnumDeclaration()) {
    auto et = CreateEnumType(ed->getType()->isTypeEnum());
    if (llvm::isa<llvm::DICompositeType>(et))
      return et;
    return EmitNamespace(ed, ed->toChars());
  } else {
    error(parent->loc, "unknown debuginfo scope `%s`; please file an LDC issue",
          parent->toChars());
    fatal();
  }
}

DIScope DIBuilder::GetCurrentScope() {
  if (IR->funcGenStates.empty())
    return getIrModule(IR->dmodule)->diModule;
  IrFunction *fn = IR->func();
  if (fn->diLexicalBlocks.empty()) {
    assert(static_cast<llvm::MDNode *>(fn->diSubprogram) != 0);
    return fn->diSubprogram;
  }
  return fn->diLexicalBlocks.top();
}

// Usually just returns the regular name of the symbol and sets the scope
// representing its parent.
// As a special case, it handles `TemplatedSymbol!(...).TemplatedSymbol`,
// returning `TemplatedSymbol!(...)` as name and the scope of the
// TemplateInstance instead.
llvm::StringRef DIBuilder::GetNameAndScope(Dsymbol *sym, DIScope &scope) {
  llvm::StringRef name;
  scope = nullptr;

  if (auto ti = sym->parent->isTemplateInstance()) {
    if (ti->aliasdecl == sym) {
      name = getTemplateInstanceName(ti);
      scope = GetSymbolScope(ti);
    }
  }

  // normal case
  if (!scope) {
    name = sym->toChars();
    scope = GetSymbolScope(sym);
  }

  return processDIName(name);
}

// Sets the memory address for a debuginfo variable.
void DIBuilder::Declare(const Loc &loc, llvm::Value *storage,
                        DILocalVariable divar, DIExpression diexpr) {
  auto debugLoc = llvm::DILocation::get(IR->context(), loc.linnum(),
                                        getColumn(loc), GetCurrentScope());
  DBuilder.insertDeclare(storage, divar, diexpr, debugLoc, IR->scopebb());
}

// Sets the (current) value for a debuginfo variable.
void DIBuilder::SetValue(const Loc &loc, llvm::Value *value,
                         DILocalVariable divar, DIExpression diexpr) {
  auto debugLoc = llvm::DILocation::get(IR->context(), loc.linnum(),
                                        getColumn(loc), GetCurrentScope());
  DBuilder.insertDbgValueIntrinsic(value, divar, diexpr, debugLoc,
                                   IR->scopebb());
}

DIFile DIBuilder::CreateFile(const char *filename) {
  if (!filename)
    filename = IR->dmodule->srcfile.toChars();

  // clang appears to use the curent working dir as 'directory' for relative
  // source paths, and the root path for absolute ones:
  // clang -g -emit-llvm -S ..\blub.c =>
  //   !DIFile(filename: "..\\blub.c", directory: "C:\\LDC\\ninja-ldc", ...)
  //   !DIFile(filename: "Program
  //   Files\\LLVM\\lib\\clang\\11.0.1\\include\\stddef.h", directory: "C:\\",
  //   ...)

  if (llvm::sys::path::is_absolute(filename)) {
    return DBuilder.createFile(llvm::sys::path::relative_path(filename),
                               llvm::sys::path::root_path(filename));
  }

  llvm::SmallString<128> cwd;
  llvm::sys::fs::current_path(cwd);

  return DBuilder.createFile(filename, cwd);
}

DIFile DIBuilder::CreateFile(const Loc &loc) {
  return CreateFile(loc.filename());
}

DIFile DIBuilder::CreateFile(Dsymbol *decl) {
  const char *filename = nullptr;
  for (Dsymbol *sym = decl; sym && !filename; sym = sym->parent)
    filename = sym->loc.filename();
  return CreateFile(filename);
}

DIType DIBuilder::CreateBasicType(Type *type) {
  using namespace llvm::dwarf;

  Type *t = type->toBasetype();
  llvm::Type *T = DtoType(type);

  // find encoding
  unsigned Encoding;
  switch (t->ty) {
  case TY::Tbool:
    Encoding = DW_ATE_boolean;
    break;
  case TY::Tchar:
    if (emitCodeView) {
      // VS debugger does not support DW_ATE_UTF for char
      Encoding = DW_ATE_unsigned_char;
      break;
    }
    // fall through
  case TY::Twchar:
  case TY::Tdchar:
    Encoding = DW_ATE_UTF;
    break;
  case TY::Tint8:
    if (emitCodeView) {
      // VS debugger does not support DW_ATE_signed for 8-bit
      Encoding = DW_ATE_signed_char;
      break;
    }
    // fall through
  case TY::Tint16:
  case TY::Tint32:
  case TY::Tint64:
  case TY::Tint128:
    Encoding = DW_ATE_signed;
    break;
  case TY::Tuns8:
    if (emitCodeView) {
      // VS debugger does not support DW_ATE_unsigned for 8-bit
      Encoding = DW_ATE_unsigned_char;
      break;
    }
    // fall through
  case TY::Tuns16:
  case TY::Tuns32:
  case TY::Tuns64:
  case TY::Tuns128:
    Encoding = DW_ATE_unsigned;
    break;
  case TY::Tfloat32:
  case TY::Tfloat64:
  case TY::Tfloat80:
    Encoding = DW_ATE_float;
    break;
  case TY::Timaginary32:
  case TY::Timaginary64:
  case TY::Timaginary80:
    if (emitCodeView) {
      // DW_ATE_imaginary_float not supported by the LLVM DWARF->CodeView
      // conversion
      Encoding = DW_ATE_float;
      break;
    }
    Encoding = DW_ATE_imaginary_float;
    break;
  case TY::Tcomplex32:
  case TY::Tcomplex64:
  case TY::Tcomplex80:
    if (emitCodeView) {
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
                                  Encoding);
}

DIType DIBuilder::CreateEnumType(TypeEnum *type) {
  EnumDeclaration *const ed = type->sym;

  if (!ed->memtype) // opaque enum
    return CreateUnspecifiedType(ed);

  if (ed->isSpecial()) // magic enums: forward to base type
    return CreateTypeDescription(ed->memtype);

  DIScope scope = nullptr;
  const auto name = GetNameAndScope(ed, scope);
  const auto lineNumber = ed->loc.linnum();
  const auto file = CreateFile(ed);

  // just emit a typedef for non-integral base types
  auto tb = type->toBasetype();
  if (!tb->isintegral()) {
    auto tbase = CreateTypeDescription(tb);
    return DBuilder.createTypedef(tbase, name, file, lineNumber, scope);
  }

  llvm::SmallVector<LLMetadata *, 8> subscripts;
  if (ed->members) {
    for (auto m : *ed->members) {
      EnumMember *em = m->isEnumMember();
      if (auto ie = em->value()->isIntegerExp()) {
        subscripts.push_back(
            DBuilder.createEnumerator(em->toChars(), ie->toInteger()));
      }
    }
  }

  llvm::Type *const T = DtoType(type);
  return DBuilder.createEnumerationType(
      scope, name, file, lineNumber,
      getTypeAllocSize(T) * 8,               // size (bits)
      getABITypeAlign(T) * 8,                // align (bits)
      DBuilder.getOrCreateArray(subscripts), // subscripts
      CreateTypeDescription(ed->memtype));
}

DIType DIBuilder::CreatePointerType(TypePointer *type) {
  // TODO: The addressspace is important for dcompute targets. See e.g.
  // https://www.mail-archive.com/dwarf-discuss@lists.dwarfstd.org/msg00326.html
  const llvm::Optional<unsigned> DWARFAddressSpace = llvm::None;

  const auto name = processDIName(type->toPrettyChars(true));

  return DBuilder.createPointerType(
      CreateTypeDescription(type->nextOf(), /*voidToUbyte=*/true),
      target.ptrsize * 8, 0, DWARFAddressSpace, name);
}

DIType DIBuilder::CreateVectorType(TypeVector *type) {
  LLType *T = DtoType(type);

  const auto dim = type->basetype->isTypeSArray()->dim->toInteger();
  const auto Dim = llvm::ConstantAsMetadata::get(DtoConstSize_t(dim));
  auto subscript = DBuilder.getOrCreateSubrange(Dim, nullptr, nullptr, nullptr);

  return DBuilder.createVectorType(
      getTypeAllocSize(T) * 8, // size (bits)
      getABITypeAlign(T) * 8,  // align (bits)
      CreateTypeDescription(type->elementType(), /*voidToUbyte=*/true),
      DBuilder.getOrCreateArray({subscript}) // subscripts
  );
}

DIType DIBuilder::CreateComplexType(Type *type) {
  Type *t = type->toBasetype();
  llvm::Type *T = DtoType(type);

  Type *elemtype = nullptr;
  switch (t->ty) {
  case TY::Tcomplex32:
    elemtype = Type::tfloat32;
    break;
  case TY::Tcomplex64:
    elemtype = Type::tfloat64;
    break;
  case TY::Tcomplex80:
    elemtype = Type::tfloat80;
    break;
  default:
    llvm_unreachable(
        "Unexpected type for debug info in DIBuilder::CreateComplexType");
  }
  DIFile file = CreateFile();

  auto imoffset = getTypeAllocSize(DtoType(elemtype));
  LLMetadata *elems[] = {
      CreateMemberType(0, elemtype, file, "re", 0, Visibility::public_),
      CreateMemberType(0, elemtype, file, "im", imoffset, Visibility::public_)};

  return DBuilder.createStructType(GetCU(),
                                   t->toChars(),            // Name
                                   file,                    // File
                                   0,                       // LineNo
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment
                                   DIFlags::FlagZero,       // What here?
                                   nullptr,                 // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,               // RunTimeLang
                                   nullptr,         // VTableHolder
                                   uniqueIdent(t)); // UniqueIdentifier
}

DIType DIBuilder::CreateMemberType(unsigned linnum, Type *type, DIFile file,
                                   const char *c_name, unsigned offset,
                                   Visibility::Kind visibility, bool isStatic,
                                   DIScope scope) {
  llvm::Type *T = DtoType(type);

  // find base type
  DIType basetype = CreateTypeDescription(type);

  auto Flags = DIFlags::FlagZero;
  switch (visibility) {
  case Visibility::private_:
    Flags = DIFlags::FlagPrivate;
    break;
  case Visibility::protected_:
    Flags = DIFlags::FlagProtected;
    break;
  case Visibility::public_:
    Flags = DIFlags::FlagPublic;
    break;
  default:
    break;
  }

  if (isStatic)
    Flags |= DIFlags::FlagStaticMember;

  return DBuilder.createMemberType(scope,
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

void DIBuilder::AddFields(AggregateDeclaration *ad, DIFile file,
                          llvm::SmallVector<LLMetadata *, 16> &elems) {
  size_t narr = ad->fields.length;
  elems.reserve(narr);
  for (auto vd : ad->fields) {
    if (vd->type->toBasetype()->isTypeNoreturn())
      continue;

    elems.push_back(CreateMemberType(vd->loc.linnum(), vd->type, file,
                                     vd->toChars(), vd->offset,
                                     vd->visibility.kind));
  }
}

void DIBuilder::AddStaticMembers(AggregateDeclaration *ad, DIFile file,
                                 llvm::SmallVector<LLMetadata *, 16> &elems) {
  auto members = ad->members;
  if (!members)
    return;

  auto scope = CreateCompositeType(ad->getType());

  std::function<void(Dsymbols *)> visitMembers = [&](Dsymbols *members) {
    for (auto s : *members) {
      if (auto attrib = s->isAttribDeclaration()) {
        if (Dsymbols *d = attrib->include(nullptr))
          visitMembers(d);
      } else if (auto tmixin = s->isTemplateMixin()) {
        // FIXME: static variables inside a template mixin need to be put inside
        // a child DICompositeType for their value to become accessible
        // (mangling issue).
        // Also DWARF supports imported declarations, but LLVM
        // currently does nothing with DIImportedEntity except at CU-level.
        visitMembers(tmixin->members);
      } else if (auto vd = s->isVarDeclaration()) {
        if (vd->isDataseg()) {
          if (auto td = vd->aliasTuple) { // ugly kludge for tuples
            if (td->isexp && td->objects) {
              Dsymbols tupleVars;
              for (auto o : *td->objects) {
                if (auto e = isExpression(o))
                  if (auto ve = e->isVarExp())
                    if (auto vd2 = ve->var->isVarDeclaration())
                      if (vd2->isDataseg())
                        tupleVars.push(vd2);
              }
              visitMembers(&tupleVars);
            }
          } else if (!vd->type->toBasetype()->isTypeNoreturn()) {
            llvm::MDNode *elem =
                CreateMemberType(vd->loc.linnum(), vd->type, file,
                                 vd->toChars(), 0, vd->visibility.kind,
                                 /*isStatic = */ true, scope);
            elems.push_back(elem);
            StaticDataMemberCache[vd].reset(elem);
          }
        }
      } /*else if (auto fd = s->isFuncDeclaration())*/ // Clang also adds static
                                                       // functions as
                                                       // declarations, but they
                                                       // already work without
                                                       // adding them.
    }
  };
  visitMembers(members);
}

DIType DIBuilder::CreateCompositeType(Type *t) {
  assert((t->ty == TY::Tstruct || t->ty == TY::Tclass) &&
         "Unsupported type for debug info in DIBuilder::CreateCompositeType");

  AggregateDeclaration *ad;
  if (t->ty == TY::Tstruct) {
    ad = static_cast<TypeStruct *>(t)->sym;
  } else {
    ad = static_cast<TypeClass *>(t)->sym;
  }

  // Use the actual type associated with the declaration, ignoring any
  // const/wrappers.
  DtoType(ad->type);
  IrAggr *irAggr = getIrAggr(ad, true);
  LLType *T = irAggr->getLLStructType();

  if (irAggr->diCompositeType) {
    return irAggr->diCompositeType;
  }

  DIScope scope = nullptr;
  const auto name = GetNameAndScope(ad, scope);

  // if we don't know the aggregate's size, we don't know enough about it
  // to provide debug info. probably a forward-declared struct?
  if (ad->sizeok == Sizeok::none) {
    return CreateUnspecifiedType(ad);
  }

  assert(GetCU() && "Compilation unit missing or corrupted");

  // elements
  llvm::SmallVector<LLMetadata *, 16> elems;

  // defaults
  const auto file = CreateFile(ad);
  const auto lineNum = ad->loc.linnum();
  const auto sizeInBits = T->isSized() ? getTypeAllocSize(T) * 8 : 0;
  const auto alignmentInBits = T->isSized() ? getABITypeAlign(T) * 8 : 0;
  const auto classOffsetInBits = 0;
  DIType derivedFrom = nullptr;
  const auto vtableHolder = nullptr;
  const auto templateParams = nullptr;
  const auto uniqueIdentifier = uniqueIdent(t);

  // set diCompositeType to handle recursive types properly
  unsigned tag = (t->ty == TY::Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                        : llvm::dwarf::DW_TAG_class_type;
  irAggr->diCompositeType =
      DBuilder.createReplaceableCompositeType(tag, name, scope, file, lineNum);

  if (!ad->isInterfaceDeclaration()) // plain interfaces don't have one
  {
    ClassDeclaration *classDecl = ad->isClassDeclaration();
    if (classDecl && classDecl->baseClass) {
      derivedFrom = CreateCompositeType(classDecl->baseClass->getType());
      auto dt = DBuilder.createInheritance(irAggr->diCompositeType,
                                           derivedFrom, // base class type
                                           0,           // offset of base class
                                           0, // offset of virtual base pointer
                                           DIFlags::FlagPublic);
      elems.push_back(dt);
    }
    AddFields(ad, file, elems);
  }
  AddStaticMembers(ad, file, elems);

  const auto elemsArray = DBuilder.getOrCreateArray(elems);

  DIType ret;
  if (t->ty == TY::Tclass) {
    ret = DBuilder.createClassType(
        scope, name, file, lineNum, sizeInBits, alignmentInBits,
        classOffsetInBits, DIFlags::FlagZero, derivedFrom, elemsArray,
        vtableHolder, templateParams, uniqueIdentifier);
  } else {
    const auto runtimeLang = 0;
    ret = DBuilder.createStructType(scope, name, file, lineNum, sizeInBits,
                                    alignmentInBits, DIFlags::FlagZero,
                                    derivedFrom, elemsArray, runtimeLang,
                                    vtableHolder, uniqueIdentifier);
  }

  irAggr->diCompositeType =
      DBuilder.replaceTemporary(llvm::TempDINode(irAggr->diCompositeType), ret);

  return irAggr->diCompositeType;
}

DIType DIBuilder::CreateArrayType(TypeArray *type) {
  llvm::Type *T = DtoType(type);

  const auto scope = GetCU();
  const auto name = processDIName(type->toPrettyChars(true));
  const auto file = CreateFile();

  LLMetadata *elems[] = {CreateMemberType(0, Type::tsize_t, file, "length", 0,
                                          Visibility::public_),
                         CreateMemberType(0, type->nextOf()->pointerTo(), file,
                                          "ptr", target.ptrsize,
                                          Visibility::public_)};

  return DBuilder.createStructType(scope, name, file,
                                   0,                       // LineNo
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   DIFlags::FlagZero,       // What here?
                                   nullptr,                 // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,                  // RunTimeLang
                                   nullptr,            // VTableHolder
                                   uniqueIdent(type)); // UniqueIdentifier
}

DIType DIBuilder::CreateSArrayType(TypeSArray *type) {
  llvm::Type *T = DtoType(type);

  Type *te = type;
  llvm::SmallVector<LLMetadata *, 8> subscripts;
  for (; te->ty == TY::Tsarray; te = te->nextOf()) {
    TypeSArray *tsa = static_cast<TypeSArray *>(te);
    const auto count = tsa->dim->toInteger();
    const auto Count = llvm::ConstantAsMetadata::get(DtoConstSize_t(count));
    const auto subscript =
        DBuilder.getOrCreateSubrange(Count, nullptr, nullptr, nullptr);
    subscripts.push_back(subscript);
  }

  return DBuilder.createArrayType(
      getTypeAllocSize(T) * 8,              // size (bits)
      getABITypeAlign(T) * 8,               // align (bits)
      CreateTypeDescription(te, /*voidToUbyte=*/true),
      DBuilder.getOrCreateArray(subscripts) // subscripts
  );
}

DIType DIBuilder::CreateAArrayType(TypeAArray *type) {
  llvm::Type *T = DtoType(type);

  auto tindex = CreateTypeDescription(type->index);
  auto tvalue = CreateTypeDescription(type->nextOf());

  const auto scope = GetCU();
  const auto name = processDIName(type->toPrettyChars(true));
  const auto file = CreateFile();

  LLMetadata *elems[] = {
      DBuilder.createTypedef(tindex, "__key_t", file, 0, scope),
      DBuilder.createTypedef(tvalue, "__val_t", file, 0, scope),
      CreateMemberType(0, Type::tvoidptr, file, "ptr", 0, Visibility::public_)};

  return DBuilder.createStructType(scope, name, file,
                                   0,                       // LineNo
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   DIFlags::FlagZero,       // What here?
                                   nullptr,                 // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,                  // RunTimeLang
                                   nullptr,            // VTableHolder
                                   uniqueIdent(type)); // UniqueIdentifier
}

////////////////////////////////////////////////////////////////////////////////

DISubroutineType DIBuilder::CreateFunctionType(Type *type) {
  TypeFunction *t = type->isTypeFunction();
  assert(t);

  Type *retType = t->next;

  // Create "dummy" subroutine type for the return type
  LLMetadata *params = {CreateTypeDescription(retType)};
  auto paramsArray = DBuilder.getOrCreateTypeArray(params);

  return DBuilder.createSubroutineType(paramsArray, DIFlags::FlagZero, 0);
}

DISubroutineType DIBuilder::CreateEmptyFunctionType() {
  auto paramsArray = DBuilder.getOrCreateTypeArray(llvm::None);
  return DBuilder.createSubroutineType(paramsArray);
}

DIType DIBuilder::CreateDelegateType(TypeDelegate *type) {
  llvm::Type *T = DtoType(type);

  const auto scope = GetCU();
  const auto name = processDIName(type->toPrettyChars(true));
  const auto file = CreateFile();

  LLMetadata *elems[] = {
      CreateMemberType(0, Type::tvoidptr, file, "ptr", 0,
                       Visibility::public_),
      CreateMemberType(0, type->next->pointerTo(), file, "funcptr",
                       target.ptrsize, Visibility::public_)};

  return DBuilder.createStructType(scope, name, file,
                                   0, // line number where defined
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   DIFlags::FlagZero,       // flags
                                   nullptr,                 // derived from
                                   DBuilder.getOrCreateArray(elems),
                                   0,                  // RunTimeLang
                                   nullptr,            // VTableHolder
                                   uniqueIdent(type)); // UniqueIdentifier
}

DIType DIBuilder::CreateUnspecifiedType(Dsymbol *sym) {
  return DBuilder.createUnspecifiedType(
      processDIName(sym->toPrettyChars(true)));
}

////////////////////////////////////////////////////////////////////////////////

DIType DIBuilder::CreateTypeDescription(Type *t, bool voidToUbyte) {
  if (voidToUbyte && t->toBasetype()->ty == TY::Tvoid)
    t = Type::tuns8;

  if (t->ty == TY::Tvoid || t->ty == TY::Tnoreturn)
    return nullptr;
  if (t->ty == TY::Tnull) {
    // display null as void*
    return DBuilder.createPointerType(
        CreateTypeDescription(Type::tvoid), target.ptrsize * 8, 0,
        /* DWARFAddressSpace */ llvm::None, "typeof(null)");
  }
  if (auto te = t->isTypeEnum())
    return CreateEnumType(te);
  if (auto tv = t->isTypeVector())
    return CreateVectorType(tv);
  if (t->isintegral() || t->isfloating())
    return CreateBasicType(t);
  if (auto tp = t->isTypePointer())
    return CreatePointerType(tp);
  if (auto ta = t->isTypeDArray())
    return CreateArrayType(ta);
  if (auto tsa = t->isTypeSArray())
    return CreateSArrayType(tsa);
  if (auto taa = t->isTypeAArray())
    return CreateAArrayType(taa);
  if (t->ty == TY::Tstruct)
    return CreateCompositeType(t);
  if (auto tc = t->isTypeClass()) {
    const auto aggregateDIType = CreateCompositeType(t);
    const auto name =
        (tc->sym->toPrettyChars(true) + llvm::StringRef("*")).str();
    return DBuilder.createPointerType(aggregateDIType, target.ptrsize * 8, 0,
                                      llvm::None, processDIName(name));
  }
  if (auto tf = t->isTypeFunction())
    return CreateFunctionType(tf);
  if (auto td = t->isTypeDelegate())
    return CreateDelegateType(td);

  // Crash if the type is not supported.
  llvm_unreachable("Unsupported type in debug info");
}

////////////////////////////////////////////////////////////////////////////////

llvm::DICompileUnit::DebugEmissionKind getDebugEmissionKind() {
  switch (global.params.symdebug) {
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
}

void DIBuilder::EmitCompileUnit(Module *m) {
  if (!mustEmitLocationsDebugInfo()) {
    return;
  }

  Logger::println("D to dwarf compile_unit");
  LOG_SCOPE;

  assert(!CUNode && "Already created compile unit for this DIBuilder instance");

  // prepare producer name string
  auto producerName =
      std::string("LDC ") + ldc_version + " (LLVM " + llvm_version + ")";

  if (emitCodeView) {
    IR->module.addModuleFlag(llvm::Module::Warning, "CodeView", 1);
  } else {
    unsigned dwarfVersion = global.params.dwarfVersion;
    if (dwarfVersion == 0 &&
        global.params.targetTriple->isWindowsMSVCEnvironment()) {
      // clang 10 defaults to v4
      dwarfVersion = 4;
    }

    if (dwarfVersion > 0) {
      IR->module.addModuleFlag(llvm::Module::Warning, "Dwarf Version",
                               dwarfVersion);
    }
  }

  // Metadata without a correct version will be stripped by UpgradeDebugInfo.
  IR->module.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);

  CUNode = DBuilder.createCompileUnit(
      global.params.symdebug == 2 ? llvm::dwarf::DW_LANG_C_plus_plus
                                  : llvm::dwarf::DW_LANG_D,
      CreateFile(m->srcfile.toChars()), producerName,
      isOptimizationEnabled(), // isOptimized
      llvm::StringRef(),       // Flags TODO
      1,                       // Runtime Version TODO
      llvm::StringRef(),       // SplitName
      getDebugEmissionKind(),  // DebugEmissionKind
      0                        // DWOId
  );
}

DIModule DIBuilder::EmitModule(Module *m) {
  if (!mustEmitFullDebugInfo()) {
    return nullptr;
  }

  IrModule *irm = getIrModule(m);
  if (irm->diModule)
    return irm->diModule;

  const auto name = processDIName(m->toPrettyChars(true));

  irm->diModule = DBuilder.createModule(
      CreateFile(m->srcfile.toChars()),
      name,              // qualified module name
      llvm::StringRef(), // (clang modules specific) ConfigurationMacros
      llvm::StringRef(), // (clang modules specific) IncludePath
      llvm::StringRef()  // (clang modules specific) ISysRoot
  );

  return irm->diModule;
}

DINamespace DIBuilder::EmitNamespace(Dsymbol *sym, llvm::StringRef name) {
  name = processDIName(name);
  const bool exportSymbols = true;
  return DBuilder.createNameSpace(GetSymbolScope(sym), name, exportSymbols);
}

void DIBuilder::EmitImport(Import *im) {
  if (!mustEmitFullDebugInfo()) {
    return;
  }

  auto diModule = EmitModule(im->mod);

  DBuilder.createImportedModule(GetCurrentScope(),
                                diModule,        // imported module
                                CreateFile(im),  // file
                                im->loc.linnum() // line num
  );
}

DISubprogram DIBuilder::EmitSubProgram(FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  IrFunction *const irFunc = getIrFunc(fd);
  if (irFunc->diSubprogram)
    return irFunc->diSubprogram;

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  assert(GetCU() &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

  DIScope scope = nullptr;
  llvm::StringRef name;
  // FIXME: work around apparent LLVM CodeView bug wrt. nested functions
  if (emitCodeView && fd->toParent2()->isFuncDeclaration()) {
    // emit into module & use fully qualified name
    scope = GetCU();
    name = processDIName(fd->toPrettyChars(true));
  } else if (fd->isMain()) {
    scope = GetSymbolScope(fd);
    name = fd->toPrettyChars(true); // `D main`
  } else {
    name = GetNameAndScope(fd, scope);
  }

  const auto linkageName = irFunc->getLLVMFuncName();
  const auto file = CreateFile(fd);
  const auto lineNo = fd->loc.linnum();
  const auto isLocalToUnit = fd->visibility.kind == Visibility::private_;
  const auto isDefinition = true;
  const auto scopeLine = lineNo; // FIXME
  const auto flags = DIFlags::FlagPrototyped;
  const auto isOptimized = isOptimizationEnabled();
  const auto dispFlags =
      llvm::DISubprogram::toSPFlags(isLocalToUnit, isDefinition, isOptimized);

  DISubroutineType diFnType = nullptr;
  if (!mustEmitFullDebugInfo()) {
    diFnType = CreateEmptyFunctionType();
  } else {
    // A special case is `auto foo() { struct S{}; S s; return s; }`
    // The return type is a nested struct, so for this particular
    // chicken-and-egg case we need to create a temporary subprogram.
    irFunc->diSubprogram = DBuilder.createTempFunctionFwdDecl(
        scope, name, linkageName, file, lineNo, /*ty=*/nullptr, scopeLine,
        flags, dispFlags);

    // Now create subroutine type.
    diFnType = CreateFunctionType(fd->type);
  }

  // FIXME: duplicates?
  auto SP = CreateFunction(scope, name, linkageName, file, lineNo, diFnType,
                           isLocalToUnit, isDefinition, isOptimized, scopeLine,
                           flags);

  if (mustEmitFullDebugInfo())
    DBuilder.replaceTemporary(llvm::TempDINode(irFunc->diSubprogram), SP);

  irFunc->diSubprogram = SP;
  return SP;
}

DISubprogram DIBuilder::CreateFunction(DIScope scope, llvm::StringRef name,
                                       llvm::StringRef linkageName, DIFile file,
                                       unsigned lineNo, DISubroutineType ty,
                                       bool isLocalToUnit, bool isDefinition,
                                       bool isOptimized, unsigned scopeLine,
                                       DIFlags flags) {
  const auto dispFlags =
      llvm::DISubprogram::toSPFlags(isLocalToUnit, isDefinition, isOptimized);
  return DBuilder.createFunction(scope, name, linkageName, file, lineNo, ty,
                                 scopeLine, flags, dispFlags);
}

DISubprogram DIBuilder::EmitThunk(llvm::Function *Thunk, FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  Logger::println("Thunk to dwarf subprogram");
  LOG_SCOPE;

  assert(GetCU() &&
         "Compilation unit missing or corrupted in DIBuilder::EmitThunk");

  // Create subroutine type (thunk has same type as wrapped function)
  DISubroutineType DIFnType = CreateFunctionType(fd->type);

  const auto scope = GetSymbolScope(fd);
  const auto name = (llvm::Twine(fd->toChars()) + ".__thunk").str();
  const auto linkageName = Thunk->getName();
  const auto file = CreateFile(fd);
  const auto lineNo = fd->loc.linnum();
  const bool isLocalToUnit = fd->visibility.kind == Visibility::private_;
  const bool isDefinition = true;
  const bool isOptimized = isOptimizationEnabled();
  const auto scopeLine = lineNo; // FIXME
  const auto flags = DIFlags::FlagPrototyped;

  return CreateFunction(scope, name, linkageName, file, lineNo, DIFnType,
                        isLocalToUnit, isDefinition, isOptimized, scopeLine,
                        flags);
}

DISubprogram DIBuilder::EmitModuleCTor(llvm::Function *Fn,
                                       llvm::StringRef prettyname) {
  if (!mustEmitLocationsDebugInfo()) {
    return nullptr;
  }

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  assert(GetCU() &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");
  DIFile file = CreateFile();

  // Create "dummy" subroutine type for the return type
  LLMetadata *params = {CreateTypeDescription(Type::tvoid)};
  auto paramsArray = DBuilder.getOrCreateTypeArray(params);
  auto DIFnType = DBuilder.createSubroutineType(paramsArray);

  const auto scope = GetCurrentScope();
  const auto linkageName = Fn->getName();
  const auto lineNo = 0;
  const bool isLocalToUnit = true;
  const bool isDefinition = true;
  const bool isOptimized = isOptimizationEnabled();
  const auto scopeLine = 0; // FIXME
  const auto flags = DIFlags::FlagPrototyped | DIFlags::FlagArtificial;

  auto SP = CreateFunction(scope, prettyname, linkageName, file, lineNo,
                           DIFnType, isLocalToUnit, isDefinition, isOptimized,
                           scopeLine, flags);
  Fn->setSubprogram(SP);
  return SP;
}

void DIBuilder::EmitFuncStart(FuncDeclaration *fd) {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf funcstart");
  LOG_SCOPE;

  auto irFunc = getIrFunc(fd);
  assert(irFunc->diSubprogram);
  irFunc->getLLVMFunc()->setSubprogram(irFunc->diSubprogram);

  IR->ir->SetCurrentDebugLocation({}); // clear first
  EmitStopPoint(fd->loc);
}

void DIBuilder::EmitBlockStart(const Loc &loc) {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf block start");
  LOG_SCOPE;

  DILexicalBlock block = DBuilder.createLexicalBlock(
      GetCurrentScope(), CreateFile(loc), loc.linnum(), getColumn(loc));
  IR->func()->diLexicalBlocks.push(block);
  EmitStopPoint(loc);
}

void DIBuilder::EmitBlockEnd() {
  if (!mustEmitLocationsDebugInfo())
    return;

  Logger::println("D to dwarf block end");
  LOG_SCOPE;

  IrFunction *fn = IR->func();
  assert(!fn->diLexicalBlocks.empty());
  fn->diLexicalBlocks.pop();
}

void DIBuilder::EmitStopPoint(const Loc &loc) {
  if (!mustEmitLocationsDebugInfo())
    return;

  // If we already have a location set and the current loc is invalid
  // (line 0), then we can just ignore it (see GitHub issue #998 for why we
  // cannot do this in all cases).
  if (!loc.linnum() && IR->ir->getCurrentDebugLocation())
    return;

  unsigned linnum = loc.linnum();
  // without proper loc use the line of the enclosing symbol that has line
  // number debug info
  for (Dsymbol *sym = IR->func()->decl; sym && !linnum; sym = sym->parent)
    linnum = sym->loc.linnum();
  if (!linnum)
    linnum = 1;

  unsigned col = getColumn(loc);
  Logger::println("D to dwarf stoppoint at line %u, column %u", linnum, col);
  LOG_SCOPE;

  IR->ir->SetCurrentDebugLocation(
      llvm::DILocation::get(IR->context(), linnum, col, GetCurrentScope()));
}

void DIBuilder::EmitValue(llvm::Value *val, VarDeclaration *vd) {
  auto sub = IR->func()->variableMap.find(vd);
  if (sub == IR->func()->variableMap.end())
    return;

  DILocalVariable debugVariable = sub->second;
  if (!mustEmitFullDebugInfo() || !debugVariable)
    return;

  llvm::Instruction *instr = DBuilder.insertDbgValueIntrinsic(
      val, debugVariable, DBuilder.createExpression(),
      IR->ir->getCurrentDebugLocation(), IR->scopebb());
  instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

void DIBuilder::EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                                  Type *type, bool isThisPtr, bool forceAsLocal,
                                  bool isRefRVal,
#if LDC_LLVM_VER >= 1400
                                  llvm::ArrayRef<uint64_t> addr
#else
                                  llvm::ArrayRef<int64_t> addr
#endif
                                  ) {
  if (!mustEmitFullDebugInfo())
    return;

  Logger::println("D to dwarf local variable");
  LOG_SCOPE;

  if (vd->type->toBasetype()->isTypeNoreturn()) {
    Logger::println("of type noreturn, skip");
    return;
  }

  auto &variableMap = IR->func()->variableMap;
  auto sub = variableMap.find(vd);
  if (sub != variableMap.end())
    return; // ensure that the debug variable is created only once

  // get type description
  if (!type)
    type = vd->type;
  DIType TD = CreateTypeDescription(type);
  if (static_cast<llvm::MDNode *>(TD) == nullptr)
    return; // unsupported

  const bool isRefOrOut = vd->isRef() || vd->isOut(); // incl. special-ref vars
  const bool isParameter = vd->isParameter();

  // For MSVC x64, some by-value parameters need to be declared as DI locals to
  // work around garbage for both cdb and VS debuggers.
  if (emitCodeView && isParameter && !forceAsLocal && !isRefOrOut &&
      global.params.targetTriple->isArch64Bit()) {
    // 1) params rewritten by IndirectByvalRewrite
    if (isaArgument(ll) && addr.empty()) {
      forceAsLocal = true;
    } else {
      // 2) dynamic arrays, delegates and vectors
      TY ty = type->toBasetype()->ty;
      if (ty == TY::Tarray || ty == TY::Tdelegate || ty == TY::Tvector)
        forceAsLocal = true;
    }
  }

  bool useDbgValueIntrinsic = false;
  if (isRefOrOut) {
    // DW_TAG_reference_type sounds like the correct tag for `this`, but member
    // function calls won't work with GDB unless `this` gets declared as
    // DW_TAG_pointer_type.
    // This matches what GDC and Clang do.
    auto Tag = isThisPtr ? llvm::dwarf::DW_TAG_pointer_type
                         : llvm::dwarf::DW_TAG_reference_type;

    // With the exception of special-ref loop variables, the reference/pointer
    // itself is constant. So we don't have to attach the debug information to a
    // memory location and can use llvm.dbg.value to set the constant pointer
    // for the DI reference.
    useDbgValueIntrinsic = !isSpecialRefVar(vd) && isRefRVal;
    // Note: createReferenceType expects the size to be the size of a pointer,
    // not the size of the type the reference refers to.
    TD = DBuilder.createReferenceType(
        Tag, TD,
        gDataLayout->getPointerSizeInBits(), // size (bits)
        DtoAlignment(type) * 8);             // align (bits)
  }

  // get variable description
  assert(!vd->isDataseg() && "static variable");

  const auto scope = GetCurrentScope();
  const auto name = vd->toChars();
  const auto file = CreateFile(vd);
  const auto lineNum = vd->loc.linnum();
  const auto preserve = true;
  auto flags = !isThisPtr
                   ? DIFlags::FlagZero
                   : DIFlags::FlagArtificial | DIFlags::FlagObjectPointer;

  DILocalVariable debugVariable;
  if (!forceAsLocal && isParameter) {
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

    debugVariable = DBuilder.createParameterVariable(
        scope, name, argNo + 1, file, lineNum, TD, preserve, flags);
  } else {
    debugVariable = DBuilder.createAutoVariable(scope, name, file, lineNum, TD,
                                                preserve, flags);
  }
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

void DIBuilder::EmitGlobalVariable(llvm::GlobalVariable *llVar,
                                   VarDeclaration *vd) {
  if (!mustEmitFullDebugInfo())
    return;

  Logger::println("D to dwarf global_variable");
  LOG_SCOPE;

  if (vd->type->toBasetype()->isTypeNoreturn()) {
    Logger::println("of type noreturn, skip");
    return;
  }

  assert(vd->isDataseg() ||
         (vd->storage_class & (STCconst | STCimmutable) && vd->_init));

  DIScope scope = GetSymbolScope(vd);
  llvm::MDNode *Decl = nullptr;

  if (vd->isDataseg() && vd->toParent()->isAggregateDeclaration()) {
    // static aggregate member
    Decl = StaticDataMemberCache[vd];
    assert(Decl && "static aggregate member not declared");
  }

  OutBuffer mangleBuf;
  mangleToBuffer(vd, &mangleBuf);

  auto DIVar = DBuilder.createGlobalVariableExpression(
      scope,                                 // context
      vd->toChars(),                         // name
      mangleBuf.peekChars(),                 // linkage name
      CreateFile(vd),                        // file
      vd->loc.linnum(),                        // line num
      CreateTypeDescription(vd->type),       // type
      vd->visibility.kind == Visibility::private_, // is local to unit
      !(vd->storage_class & STCextern),      // bool isDefined
      nullptr,                               // DIExpression *Expr
      Decl                                   // declaration
  );

  llVar->addDebugInfo(DIVar);
}

void DIBuilder::Finalize() {
  if (!mustEmitLocationsDebugInfo())
    return;

  DBuilder.finalize();
}

} // namespace ldc
