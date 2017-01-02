//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dibuilder.h"

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

#if LDC_LLVM_VER >= 307
typedef llvm::DINode DIFlags;
#else
typedef llvm::DIDescriptor DIFlags;
#endif

////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current
// module
Module *ldc::DIBuilder::getDefinedModule(Dsymbol *s) {
  // templates are defined in current module
  if (DtoIsTemplateInstance(s)) {
    return IR->dmodule;
  }
  // array operations as well
  if (FuncDeclaration *fd = s->isFuncDeclaration()) {
    if (fd->isArrayOp && (willInline() || !isDruntimeArrayOp(fd))) {
      return IR->dmodule;
    }
  }
  // otherwise use the symbol's module
  return s->getModule();
}

////////////////////////////////////////////////////////////////////////////////

ldc::DIBuilder::DIBuilder(IRState *const IR)
    : IR(IR), DBuilder(IR->module), CUNode(nullptr) {}

llvm::LLVMContext &ldc::DIBuilder::getContext() { return IR->context(); }

ldc::DIScope ldc::DIBuilder::GetCurrentScope() {
  IrFunction *fn = IR->func();
  if (fn->diLexicalBlocks.empty()) {
    assert(static_cast<llvm::MDNode *>(fn->diSubprogram) != 0);
    return fn->diSubprogram;
  }
  return fn->diLexicalBlocks.top();
}

void ldc::DIBuilder::Declare(const Loc &loc, llvm::Value *var,
                             ldc::DILocalVariable divar
#if LDC_LLVM_VER >= 306
                             ,
                             ldc::DIExpression diexpr
#endif
                             ) {
  unsigned charnum = (loc.linnum ? loc.charnum : 0);
#if LDC_LLVM_VER < 307
  llvm::Instruction *instr = DBuilder.insertDeclare(var, divar,
#if LDC_LLVM_VER >= 306
                                                    diexpr,
#endif
                                                    IR->scopebb());
  instr->setDebugLoc(
      llvm::DebugLoc::get(loc.linnum, charnum, GetCurrentScope()));
#else // if LLVM >= 3.7
  DBuilder.insertDeclare(
      var, divar, diexpr,
      llvm::DebugLoc::get(loc.linnum, charnum, GetCurrentScope()),
      IR->scopebb());
#endif
}

ldc::DIFile ldc::DIBuilder::CreateFile(Loc &loc) {
  llvm::SmallString<128> path(loc.filename ? loc.filename : "");
  llvm::sys::fs::make_absolute(path);

  return DBuilder.createFile(llvm::sys::path::filename(path),
                             llvm::sys::path::parent_path(path));
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
  case Twchar:
  case Tdchar:
    Encoding = type->isunsigned() ? DW_ATE_unsigned_char : DW_ATE_signed_char;
    break;
  case Tint8:
  case Tint16:
  case Tint32:
  case Tint64:
  case Tint128:
    Encoding = DW_ATE_signed;
    break;
  case Tuns8:
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
    Encoding = DW_ATE_imaginary_float;
    break;
  case Tcomplex32:
  case Tcomplex64:
  case Tcomplex80:
    Encoding = DW_ATE_complex_float;
    break;
  default:
    llvm_unreachable(
        "Unsupported basic type for debug info in DIBuilder::CreateBasicType");
  }

  return DBuilder.createBasicType(type->toChars(),         // name
                                  getTypeAllocSize(T) * 8, // size (bits)
                                  getABITypeAlign(T) * 8,  // align (bits)
                                  Encoding);
}

ldc::DIType ldc::DIBuilder::CreateEnumType(Type *type) {
  llvm::Type *T = DtoType(type);

  assert(type->ty == Tenum &&
         "only enums allowed for debug info in dwarfEnumType");
  TypeEnum *te = static_cast<TypeEnum *>(type);

#if LDC_LLVM_VER >= 306
  llvm::SmallVector<llvm::Metadata *, 8> subscripts;
#else
  llvm::SmallVector<llvm::Value *, 8> subscripts;
#endif
  for (auto m : *te->sym->members) {
    EnumMember *em = m->isEnumMember();
    llvm::StringRef Name(em->toChars());
    uint64_t Val = em->value()->toInteger();
    auto Subscript = DBuilder.createEnumerator(Name, Val);
    subscripts.push_back(Subscript);
  }

  llvm::StringRef Name = te->toChars();
  unsigned LineNumber = te->sym->loc.linnum;
  ldc::DIFile File(CreateFile(te->sym->loc));

  return DBuilder.createEnumerationType(
      GetCU(), Name, File, LineNumber,
      getTypeAllocSize(T) * 8,               // size (bits)
      getABITypeAlign(T) * 8,                // align (bits)
      DBuilder.getOrCreateArray(subscripts), // subscripts
      CreateTypeDescription(te->sym->memtype, false));
}

ldc::DIType ldc::DIBuilder::CreatePointerType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();

  assert(
      t->ty == Tpointer &&
      "Only pointers allowed for debug info in DIBuilder::CreatePointerType");

  // find base type
  Type *nt = t->nextOf();
  ldc::DIType basetype(CreateTypeDescription(nt, false));

  return DBuilder.createPointerType(basetype,
                                    getTypeAllocSize(T) * 8, // size (bits)
                                    getABITypeAlign(T) * 8,  // align (bits)
                                    type->toChars()          // name
                                    );
}

ldc::DIType ldc::DIBuilder::CreateVectorType(Type *type) {
  LLType *T = DtoType(type);
  Type *t = type->toBasetype();

  assert(t->ty == Tvector &&
         "Only vectors allowed for debug info in DIBuilder::CreateVectorType");
  TypeVector *tv = static_cast<TypeVector *>(t);
  Type *te = tv->elementType();
  int64_t Dim = tv->size(Loc()) / te->size(Loc());
#if LDC_LLVM_VER >= 306
  llvm::Metadata *subscripts[] =
#else
  llvm::Value *subscripts[] =
#endif
      {DBuilder.getOrCreateSubrange(0, Dim)};
  ldc::DIType basetype(CreateTypeDescription(te, false));

  return DBuilder.createVectorType(
      getTypeAllocSize(T) * 8,              // size (bits)
      getABITypeAlign(T) * 8,               // align (bits)
      basetype,                             // element type
      DBuilder.getOrCreateArray(subscripts) // subscripts
      );
}

ldc::DIType ldc::DIBuilder::CreateMemberType(unsigned linnum, Type *type,
                                             ldc::DIFile file,
                                             const char *c_name,
                                             unsigned offset, PROTKIND prot) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();

  // find base type
  ldc::DIType basetype(CreateTypeDescription(t, true));

  unsigned Flags = 0;
  switch (prot) {
  case PROTprivate:
    Flags = DIFlags::FlagPrivate;
    break;
  case PROTprotected:
    Flags = DIFlags::FlagProtected;
    break;
#if LDC_LLVM_VER >= 306
  case PROTpublic:
    Flags = DIFlags::FlagPublic;
    break;
#endif
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

void ldc::DIBuilder::AddBaseFields(ClassDeclaration *sd, ldc::DIFile file,
#if LDC_LLVM_VER >= 306
                                   std::vector<llvm::Metadata *> &elems
#else
                                   std::vector<llvm::Value *> &elems
#endif
                                   ) {
  if (sd->baseClass) {
    AddBaseFields(sd->baseClass, file, elems);
  }

  size_t narr = sd->fields.dim;
  elems.reserve(narr);
  for (auto vd : sd->fields) {
    elems.push_back(CreateMemberType(vd->loc.linnum, vd->type, file,
                                     vd->toChars(), vd->offset,
                                     vd->prot().kind));
  }
}

ldc::DIType ldc::DIBuilder::CreateCompositeType(Type *type) {
  Type *t = type->toBasetype();
  assert((t->ty == Tstruct || t->ty == Tclass) &&
         "Unsupported type for debug info in DIBuilder::CreateCompositeType");
  AggregateDeclaration *sd;
  if (t->ty == Tstruct) {
    TypeStruct *ts = static_cast<TypeStruct *>(t);
    sd = ts->sym;
  } else {
    TypeClass *tc = static_cast<TypeClass *>(t);
    sd = tc->sym;
  }
  assert(sd);

  // Use the actual type associated with the declaration, ignoring any
  // const/wrappers.
  LLType *T = DtoType(sd->type);
  IrTypeAggr *ir = sd->type->ctype->isAggr();
  assert(ir);

  if (static_cast<llvm::MDNode *>(ir->diCompositeType) != nullptr) {
    return ir->diCompositeType;
  }

  // if we don't know the aggregate's size, we don't know enough about it
  // to provide debug info. probably a forward-declared struct?
  if (sd->sizeok == SIZEOKnone) {
    return DBuilder.createUnspecifiedType(sd->toChars());
  }

// elements
#if LDC_LLVM_VER >= 306
  std::vector<llvm::Metadata *> elems;
#else
  std::vector<llvm::Value *> elems;
#endif

  // defaults
  llvm::StringRef name = sd->toChars();
  unsigned linnum = sd->loc.linnum;
  ldc::DICompileUnit CU(GetCU());
  assert(CU && "Compilation unit missing or corrupted");
  ldc::DIFile file(CreateFile(sd->loc));
#if LDC_LLVM_VER >= 307
  ldc::DIType derivedFrom = nullptr;
#else
  ldc::DIType derivedFrom;
#endif

  // set diCompositeType to handle recursive types properly
  unsigned tag = (t->ty == Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                    : llvm::dwarf::DW_TAG_class_type;
#if LDC_LLVM_VER >= 307
  ir->diCompositeType = DBuilder.createReplaceableCompositeType(
#else
  ir->diCompositeType = DBuilder.createReplaceableForwardDecl(
#endif
      tag, name, CU, file, linnum);

  if (!sd->isInterfaceDeclaration()) // plain interfaces don't have one
  {
    if (t->ty == Tstruct) {
      elems.reserve(sd->fields.dim);
      for (auto vd : sd->fields) {
        ldc::DIType dt =
            CreateMemberType(vd->loc.linnum, vd->type, file, vd->toChars(),
                             vd->offset, vd->prot().kind);
        elems.push_back(dt);
      }
    } else {
      ClassDeclaration *classDecl = sd->isClassDeclaration();
      AddBaseFields(classDecl, file, elems);
      if (classDecl->baseClass) {
        derivedFrom = CreateCompositeType(classDecl->baseClass->getType());
      }
    }
  }

#if LDC_LLVM_VER >= 307
  llvm::DINodeArray elemsArray = DBuilder.getOrCreateArray(elems);
#else
  llvm::DIArray elemsArray = DBuilder.getOrCreateArray(elems);
#endif

  ldc::DIType ret;
  if (t->ty == Tclass) {
    ret = DBuilder.createClassType(CU,     // compile unit where defined
                                   name,   // name
                                   file,   // file where defined
                                   linnum, // line number where defined
                                   getTypeAllocSize(T) * 8, // size in bits
                                   getABITypeAlign(T) * 8,  // alignment in bits
                                   0,                       // offset in bits,
                                   DIFlags::FlagFwdDecl,    // flags
                                   derivedFrom,             // DerivedFrom
                                   elemsArray);
  } else {
    ret = DBuilder.createStructType(CU,     // compile unit where defined
                                    name,   // name
                                    file,   // file where defined
                                    linnum, // line number where defined
                                    getTypeAllocSize(T) * 8, // size in bits
                                    getABITypeAlign(T) * 8, // alignment in bits
                                    DIFlags::FlagFwdDecl,   // flags
                                    derivedFrom,            // DerivedFrom
                                    elemsArray);
  }

#if LDC_LLVM_VER >= 307
  ir->diCompositeType = DBuilder.replaceTemporary(
      llvm::TempDINode(ir->diCompositeType), static_cast<llvm::DIType *>(ret));
#else
  ir->diCompositeType.replaceAllUsesWith(ret);
#endif
  ir->diCompositeType = ret;

  return ret;
}

ldc::DIType ldc::DIBuilder::CreateArrayType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();

  assert(t->ty == Tarray &&
         "Only arrays allowed for debug info in DIBuilder::CreateArrayType");

  Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
  ldc::DIFile file(CreateFile(loc));

#if LDC_LLVM_VER >= 306
  llvm::Metadata *elems[] =
#else
  llvm::Value *elems[] =
#endif
      {CreateMemberType(0, Type::tsize_t, file, "length", 0, PROTpublic),
       CreateMemberType(0, t->nextOf()->pointerTo(), file, "ptr",
                        global.params.is64bit ? 8 : 4, PROTpublic)};

  return DBuilder.createStructType(
      GetCU(),
      llvm::StringRef(), // Name TODO: Really no name for arrays? t->toChars()?
      file,              // File
      0,                 // LineNo
      getTypeAllocSize(T) * 8, // size in bits
      getABITypeAlign(T) * 8,  // alignment in bits
      0,                       // What here?
#if LDC_LLVM_VER >= 307
      nullptr, // DerivedFrom
#else
      llvm::DIType(), // DerivedFrom
#endif
      DBuilder.getOrCreateArray(elems));
}

ldc::DIType ldc::DIBuilder::CreateSArrayType(Type *type) {
  llvm::Type *T = DtoType(type);
  Type *t = type->toBasetype();

// find base type
#if LDC_LLVM_VER >= 306
  llvm::SmallVector<llvm::Metadata *, 8> subscripts;
#else
  llvm::SmallVector<llvm::Value *, 8> subscripts;
#endif
  while (t->ty == Tsarray) {
    TypeSArray *tsa = static_cast<TypeSArray *>(t);
    int64_t Count = tsa->dim->toInteger();
#if LDC_LLVM_VER >= 306
    llvm::Metadata *subscript = DBuilder.getOrCreateSubrange(0, Count - 1);
#else
    llvm::Value *subscript = DBuilder.getOrCreateSubrange(0, Count - 1);
#endif
    subscripts.push_back(subscript);
    t = t->nextOf();
  }
  ldc::DIType basetype(CreateTypeDescription(t, false));

  return DBuilder.createArrayType(
      getTypeAllocSize(T) * 8,              // size (bits)
      getABITypeAlign(T) * 8,               // align (bits)
      basetype,                             // element type
      DBuilder.getOrCreateArray(subscripts) // subscripts
      );
}

ldc::DIType ldc::DIBuilder::CreateAArrayType(Type *type) {
  // FIXME: Implement
  return DBuilder.createUnspecifiedType(type->toChars());
}

////////////////////////////////////////////////////////////////////////////////

ldc::DISubroutineType ldc::DIBuilder::CreateFunctionType(Type *type) {
  TypeFunction *t = static_cast<TypeFunction *>(type);
  Type *retType = t->next;

// Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER == 305
  llvm::SmallVector<llvm::Value *, 16> Elts;
  auto EltTypeArray = DBuilder.getOrCreateArray(Elts);
#else
  llvm::SmallVector<llvm::Metadata *, 16> Elts;
  auto EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#endif
  Elts.push_back(CreateTypeDescription(retType, true));

#if LDC_LLVM_VER >= 308
  return DBuilder.createSubroutineType(EltTypeArray);
#else
  Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
  ldc::DIFile file(CreateFile(loc));
  return DBuilder.createSubroutineType(file, EltTypeArray);
#endif
}

ldc::DISubroutineType ldc::DIBuilder::CreateDelegateType(Type *type) {
  // FIXME: Implement
  TypeDelegate *t = static_cast<TypeDelegate *>(type);

// Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER >= 306
  llvm::SmallVector<llvm::Metadata *, 16> Elts;
#else
  llvm::SmallVector<llvm::Value *, 16> Elts;
#endif
  Elts.push_back(DBuilder.createUnspecifiedType(type->toChars()));
#if LDC_LLVM_VER >= 306
  auto EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#else
  auto EltTypeArray = DBuilder.getOrCreateArray(Elts);
#endif

#if LDC_LLVM_VER >= 308
  return DBuilder.createSubroutineType(EltTypeArray);
#else
  Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
  ldc::DIFile file(CreateFile(loc));
  return DBuilder.createSubroutineType(file, EltTypeArray);
#endif
}

////////////////////////////////////////////////////////////////////////////////
bool isOpaqueEnumType(Type *type) {
  if (type->ty != Tenum)
    return false;

  TypeEnum *te = static_cast<TypeEnum *>(type);
  return !te->sym->memtype;
}

ldc::DIType ldc::DIBuilder::CreateTypeDescription(Type *type, bool derefclass) {
  // Check for opaque enum first, Bugzilla 13792
  if (isOpaqueEnumType(type)) {
    return DBuilder.createUnspecifiedType(type->toChars());
  }

  Type *t = type->toBasetype();
  if (derefclass && t->ty == Tclass) {
    type = type->pointerTo();
    t = type->toBasetype();
  }

  if (t->ty == Tvoid || t->ty == Tnull) {
    return DBuilder.createUnspecifiedType(t->toChars());
  }
  if (t->isintegral() || t->isfloating()) {
    if (t->ty == Tvector) {
      return CreateVectorType(type);
    }
    if (type->ty == Tenum) {
      return CreateEnumType(type);
    }
    return CreateBasicType(type);
  }
  if (t->ty == Tpointer) {
    return CreatePointerType(type);
  }
  if (t->ty == Tarray) {
    return CreateArrayType(type);
  }
  if (t->ty == Tsarray) {
    return CreateSArrayType(type);
  }
  if (t->ty == Taarray) {
    return CreateAArrayType(type);
  }
  if (t->ty == Tstruct || t->ty == Tclass) {
    return CreateCompositeType(type);
  }
  if (t->ty == Tfunction) {
    return CreateFunctionType(type);
  }
  if (t->ty == Tdelegate) {
    return CreateDelegateType(type);
  }

  // Crash if the type is not supported.
  llvm_unreachable("Unsupported type in debug info");
}

////////////////////////////////////////////////////////////////////////////////

void ldc::DIBuilder::EmitCompileUnit(Module *m) {
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf compile_unit");
  LOG_SCOPE;

  assert(!CUNode && "Already created compile unit for this DIBuilder instance");

  // prepare srcpath
  llvm::SmallString<128> srcpath(m->srcfile->name->toChars());
  llvm::sys::fs::make_absolute(srcpath);

#if LDC_LLVM_VER >= 308
  if (global.params.targetTriple->isWindowsMSVCEnvironment())
    IR->module.addModuleFlag(llvm::Module::Warning, "CodeView", 1);
#endif
  // Metadata without a correct version will be stripped by UpgradeDebugInfo.
  IR->module.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);

  CUNode = DBuilder.createCompileUnit(
      global.params.symdebug == 2 ? llvm::dwarf::DW_LANG_C
                                  : llvm::dwarf::DW_LANG_D,
      llvm::sys::path::filename(srcpath), llvm::sys::path::parent_path(srcpath),
      "LDC (http://wiki.dlang.org/LDC)",
      isOptimizationEnabled(), // isOptimized
      llvm::StringRef(),       // Flags TODO
      1                        // Runtime Version TODO
      );
}

ldc::DISubprogram ldc::DIBuilder::EmitSubProgram(FuncDeclaration *fd) {
  if (!global.params.symdebug) {
#if LDC_LLVM_VER >= 307
    return nullptr;
#else
    return llvm::DISubprogram();
#endif
  }

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

  ldc::DIFile file(CreateFile(fd->loc));

  // Create subroutine type
  ldc::DISubroutineType DIFnType =
    CreateFunctionType(static_cast<TypeFunction *>(fd->type));

  // FIXME: duplicates?
  auto SP = DBuilder.createFunction(
      CU,                                 // context
      fd->toPrettyChars(),                // name
      getIrFunc(fd)->func->getName(),     // linkage name
      file,                               // file
      fd->loc.linnum,                     // line no
      DIFnType,                           // type
      fd->protection.kind == PROTprivate, // is local to unit
      true,                               // isdefinition
      fd->loc.linnum,                     // FIXME: scope line
      DIFlags::FlagPrototyped,            // Flags
      isOptimizationEnabled()             // isOptimized
#if LDC_LLVM_VER < 308
      ,
      getIrFunc(fd)->func
#endif
      );
#if LDC_LLVM_VER >= 308
  if (fd->fbody) {
    getIrFunc(fd)->func->setSubprogram(SP);
  }
#endif
  return SP;
}

ldc::DISubprogram ldc::DIBuilder::EmitThunk(llvm::Function *Thunk,
                                            FuncDeclaration *fd) {
  if (!global.params.symdebug) {
#if LDC_LLVM_VER >= 307
    return nullptr;
#else
    return llvm::DISubprogram();
#endif
  }

  Logger::println("Thunk to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU &&
         "Compilation unit missing or corrupted in DIBuilder::EmitThunk");

  ldc::DIFile file(CreateFile(fd->loc));

  // Create subroutine type (thunk has same type as wrapped function)
  ldc::DISubroutineType DIFnType = CreateFunctionType(fd->type);

  std::string name = fd->toPrettyChars();
  name.append(".__thunk");

  // FIXME: duplicates?
  auto SP = DBuilder.createFunction(
      CU,                                 // context
      name,                               // name
      Thunk->getName(),                   // linkage name
      file,                               // file
      fd->loc.linnum,                     // line no
      DIFnType,                           // type
      fd->protection.kind == PROTprivate, // is local to unit
      true,                               // isdefinition
      fd->loc.linnum,                     // FIXME: scope line
      DIFlags::FlagPrototyped,            // Flags
      isOptimizationEnabled()             // isOptimized
#if LDC_LLVM_VER < 308
      ,
      getIrFunc(fd)->func
#endif
      );
#if LDC_LLVM_VER >= 308
  if (fd->fbody) {
    getIrFunc(fd)->func->setSubprogram(SP);
  }
#endif
  return SP;
}

ldc::DISubprogram ldc::DIBuilder::EmitModuleCTor(llvm::Function *Fn,
                                                 llvm::StringRef prettyname) {
  if (!global.params.symdebug) {
#if LDC_LLVM_VER >= 307
    return nullptr;
#else
    return llvm::DISubprogram();
#endif
  }

  Logger::println("D to dwarf subprogram");
  LOG_SCOPE;

  ldc::DICompileUnit CU(GetCU());
  assert(CU &&
         "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

  Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
  ldc::DIFile file(CreateFile(loc));

// Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER >= 306
  llvm::SmallVector<llvm::Metadata *, 1> Elts;
#else
  llvm::SmallVector<llvm::Value *, 1> Elts;
#endif
  Elts.push_back(CreateTypeDescription(Type::tvoid, true));
#if LDC_LLVM_VER >= 307
  llvm::DITypeRefArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#elif LDC_LLVM_VER >= 306
  llvm::DITypeArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#else
  llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
#endif
#if LDC_LLVM_VER >= 308
  ldc::DISubroutineType DIFnType = DBuilder.createSubroutineType(EltTypeArray);
#else
  ldc::DISubroutineType DIFnType =
      DBuilder.createSubroutineType(file, EltTypeArray);
#endif

  // FIXME: duplicates?
  auto SP = DBuilder.createFunction(
      CU,            // context
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
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf funcstart");
  LOG_SCOPE;

  assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
  EmitStopPoint(fd->loc);
}

void ldc::DIBuilder::EmitFuncEnd(FuncDeclaration *fd) {
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf funcend");
  LOG_SCOPE;

  assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
  EmitStopPoint(fd->endloc);
}

void ldc::DIBuilder::EmitBlockStart(Loc &loc) {
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf block start");
  LOG_SCOPE;

  ldc::DILexicalBlock block =
      DBuilder.createLexicalBlock(GetCurrentScope(),           // scope
                                  CreateFile(loc),             // file
                                  loc.linnum,                  // line
                                  loc.linnum ? loc.charnum : 0 // column
#if LDC_LLVM_VER == 305
                                  ,
                                  0 // DWARF path discriminator value
#endif
                                  );
  IR->func()->diLexicalBlocks.push(block);
  EmitStopPoint(loc);
}

void ldc::DIBuilder::EmitBlockEnd() {
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf block end");
  LOG_SCOPE;

  IrFunction *fn = IR->func();
  assert(!fn->diLexicalBlocks.empty());
  fn->diLexicalBlocks.pop();
}

void ldc::DIBuilder::EmitStopPoint(Loc &loc) {
  if (!global.params.symdebug) {
    return;
  }

// If we already have a location set and the current loc is invalid
// (line 0), then we can just ignore it (see GitHub issue #998 for why we
// cannot do this in all cases).
#if LDC_LLVM_VER >= 307
  if (!loc.linnum && IR->ir->getCurrentDebugLocation()) {
    return;
  }
#else
  if (!loc.linnum && !IR->ir->getCurrentDebugLocation().isUnknown()) {
    return;
  }
#endif

  unsigned charnum = (loc.linnum ? loc.charnum : 0);
  Logger::println("D to dwarf stoppoint at line %u, column %u", loc.linnum,
                  charnum);
  LOG_SCOPE;
  IR->ir->SetCurrentDebugLocation(
      llvm::DebugLoc::get(loc.linnum, charnum, GetCurrentScope()));
}

void ldc::DIBuilder::EmitValue(llvm::Value *val, VarDeclaration *vd) {
  auto sub = IR->func()->variableMap.find(vd);
  if (sub == IR->func()->variableMap.end()) {
    return;
  }

  ldc::DILocalVariable debugVariable = sub->second;
  if (!global.params.symdebug || !debugVariable) {
    return;
  }

  llvm::Instruction *instr =
      DBuilder.insertDbgValueIntrinsic(val, 0, debugVariable,
#if LDC_LLVM_VER >= 306
                                       DBuilder.createExpression(),
#endif
#if LDC_LLVM_VER >= 307
                                       IR->ir->getCurrentDebugLocation(),
#endif
                                       IR->scopebb());
  instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

void ldc::DIBuilder::EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                                       Type *type, bool isThisPtr, bool fromNested,
#if LDC_LLVM_VER >= 306
                                       llvm::ArrayRef<int64_t> addr
#else
                                       llvm::ArrayRef<llvm::Value *> addr
#endif
                                       ) {
  if (!global.params.symdebug) {
    return;
  }

  Logger::println("D to dwarf local variable");
  LOG_SCOPE;

  auto &variableMap = IR->func()->variableMap;
  auto sub = variableMap.find(vd);
  if (sub != variableMap.end()) {
    return; // ensure that the debug variable is created only once
  }

  // get type description
  ldc::DIType TD = CreateTypeDescription(type ? type : vd->type, true);
  if (static_cast<llvm::MDNode *>(TD) == nullptr) {
    return; // unsupported
  }

  if (vd->storage_class & (STCref | STCout)) {
    TD = DBuilder.createReferenceType(llvm::dwarf::DW_TAG_reference_type, TD);
  }

  // get variable description
  assert(!vd->isDataseg() && "static variable");

#if LDC_LLVM_VER < 308
  unsigned tag;
  if (!fromNested && vd->isParameter()) {
    tag = llvm::dwarf::DW_TAG_arg_variable;
  } else {
    tag = llvm::dwarf::DW_TAG_auto_variable;
  }
#endif

  ldc::DILocalVariable debugVariable;
  unsigned Flags = 0;
  if (isThisPtr) {
    Flags |= DIFlags::FlagArtificial | DIFlags::FlagObjectPointer;
  }

#if LDC_LLVM_VER < 306
  if (addr.empty()) {
    debugVariable = DBuilder.createLocalVariable(tag,                 // tag
                                                 GetCurrentScope(),   // scope
                                                 vd->toChars(),       // name
                                                 CreateFile(vd->loc), // file
                                                 vd->loc.linnum, // line num
                                                 TD,             // type
                                                 true,           // preserve
                                                 Flags           // flags
                                                 );
  } else {
    debugVariable = DBuilder.createComplexVariable(tag,                 // tag
                                                   GetCurrentScope(),   // scope
                                                   vd->toChars(),       // name
                                                   CreateFile(vd->loc), // file
                                                   vd->loc.linnum, // line num
                                                   TD,             // type
                                                   addr);
  }
#elif LDC_LLVM_VER < 308
  debugVariable = DBuilder.createLocalVariable(tag,                 // tag
                                               GetCurrentScope(),   // scope
                                               vd->toChars(),       // name
                                               CreateFile(vd->loc), // file
                                               vd->loc.linnum,      // line num
                                               TD,                  // type
                                               true,                // preserve
                                               Flags                // flags
                                               );
#else
  if (!fromNested && vd->isParameter()) {
    FuncDeclaration *fd = vd->parent->isFuncDeclaration();
    assert(fd);
    size_t argNo = 0;
    if (fd->vthis != vd) {
      assert(fd->parameters);
      for (argNo = 0; argNo < fd->parameters->dim; argNo++) {
        if ((*fd->parameters)[argNo] == vd) {
          break;
        }
      }
      assert(argNo < fd->parameters->dim);
      if (fd->vthis) {
        argNo++;
      }
    }

    debugVariable =
        DBuilder.createParameterVariable(GetCurrentScope(), // scope
                                         vd->toChars(),     // name
                                         argNo + 1,
                                         CreateFile(vd->loc), // file
                                         vd->loc.linnum,      // line num
                                         TD,                  // type
                                         true,                // preserve
                                         Flags                // flags
                                         );
  } else {
    debugVariable = DBuilder.createAutoVariable(GetCurrentScope(),   // scope
                                                vd->toChars(),       // name
                                                CreateFile(vd->loc), // file
                                                vd->loc.linnum,      // line num
                                                TD,                  // type
                                                true,                // preserve
                                                Flags                // flags
                                                );
  }
#endif
  variableMap[vd] = debugVariable;

// declare
#if LDC_LLVM_VER >= 306
  Declare(vd->loc, ll, debugVariable, addr.empty()
                                          ? DBuilder.createExpression()
                                          : DBuilder.createExpression(addr));
#else
  Declare(vd->loc, ll, debugVariable);
#endif
}

ldc::DIGlobalVariable
ldc::DIBuilder::EmitGlobalVariable(llvm::GlobalVariable *ll,
                                   VarDeclaration *vd) {
  if (!global.params.symdebug) {
#if LDC_LLVM_VER >= 307
    return nullptr;
#else
    return llvm::DIGlobalVariable();
#endif
  }

  Logger::println("D to dwarf global_variable");
  LOG_SCOPE;

  assert(vd->isDataseg() ||
         (vd->storage_class & (STCconst | STCimmutable) && vd->_init));

  return DBuilder.createGlobalVariable(
#if LDC_LLVM_VER >= 306
      GetCU(), // context
#endif
      vd->toChars(),                          // name
      mangle(vd),                             // linkage name
      CreateFile(vd->loc),                    // file
      vd->loc.linnum,                         // line num
      CreateTypeDescription(vd->type, false), // type
      vd->protection.kind == PROTprivate,     // is local to unit
      ll                                      // value
      );
}

void ldc::DIBuilder::Finalize() {
  if (!global.params.symdebug) {
    return;
  }

  DBuilder.finalize();
}
