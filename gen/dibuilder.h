//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/tollvm.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Type.h"

struct IRState;

class ClassDeclaration;
class Dsymbol;
class FuncDeclaration;
class Import;
class Module;
class Type;
class VarDeclaration;

namespace llvm {
class GlobalVariable;
class StructType;
class LLVMContext;

// Only for the OpXXX templates, see below.
class DataLayout;
}

// Only for the OpXXX templates, see below.
extern const llvm::DataLayout *gDataLayout;

namespace ldc {

// Define some basic types
using DIType = llvm::DIType *;
using DICompositeType = llvm::DICompositeType *;
using DIFile = llvm::DIFile *;
using DIGlobalVariable = llvm::DIGlobalVariable *;
using DILocalVariable = llvm::DILocalVariable *;
using DIExpression = llvm::DIExpression *;
using DILexicalBlock = llvm::DILexicalBlock *;
using DINamespace = llvm::DINamespace *;
using DIScope = llvm::DIScope *;
using DISubroutineType = llvm::DISubroutineType *;
using DISubprogram = llvm::DISubprogram *;
using DIModule = llvm::DIModule *;
using DICompileUnit = llvm::DICompileUnit *;
using DIFlags = llvm::DINode::DIFlags;

class DIBuilder {
  IRState *const IR;
  llvm::DIBuilder DBuilder;

  DICompileUnit CUNode;

  const bool emitCodeView;
  const bool emitColumnInfo;

  llvm::DenseMap<Declaration*, llvm::TypedTrackingMDRef<llvm::MDNode>> StaticDataMemberCache;

  DICompileUnit GetCU() {
    return CUNode;
  }

public:
  explicit DIBuilder(IRState *const IR);

  /// \brief Emit the Dwarf compile_unit global for a Module m.
  /// \param m        Module to emit as compile unit.
  void EmitCompileUnit(Module *m);

  /// \brief Emit the Dwarf module global for a Module m.
  /// \param m        Module to emit (either as definition or declaration).
  DIModule EmitModule(Module *m);

  DINamespace EmitNamespace(Dsymbol *sym, llvm::StringRef name);

  /// \brief Emit the Dwarf imported entity and module global for an Import im.
  /// \param im        Import to emit.
  void EmitImport(Import *im);

  /// \brief Emit the Dwarf subprogram global for a function declaration fd.
  /// \param fd       Function declaration to emit as subprogram.
  /// \returns        the Dwarf subprogram global.
  DISubprogram EmitSubProgram(FuncDeclaration *fd); // FIXME

  /// \brief Emit the Dwarf subprogram global for a thunk.
  /// \param Thunk    llvm::Function pointer.
  /// \param fd       The function wrapped by this thunk.
  /// \returns        the Dwarf subprogram global.
  DISubprogram EmitThunk(llvm::Function *Thunk, FuncDeclaration *fd); // FIXME

  /// \brief Emit the Dwarf subprogram global for a module ctor.
  /// This is used for generated functions like moduleinfoctors,
  /// module ctors/dtors and unittests.
  /// \param Fn           llvm::Function pointer.
  /// \param prettyname   The name as seen in the source.
  /// \returns       the Dwarf subprogram global.
  DISubprogram EmitModuleCTor(llvm::Function *Fn,
                              llvm::StringRef prettyname); // FIXME

  /// \brief Emits debug info for function start
  void EmitFuncStart(FuncDeclaration *fd);

  /// \brief Emits debug info for block start
  void EmitBlockStart(const Loc &loc);

  /// \brief Emits debug info for block end
  void EmitBlockEnd();

  void EmitStopPoint(const Loc &loc);

  void EmitValue(llvm::Value *val, VarDeclaration *vd);

  /// \brief Emits all things necessary for making debug info for a local
  /// variable vd.
  /// \param ll       LL value which, in combination with `addr`, yields the
  /// storage/lvalue of the variable. For special-ref loop variables, specify
  /// the storage/lvalue of the reference/pointer.
  /// \param vd       Variable declaration to emit debug info for.
  /// \param type     Type of variable if different from vd->type
  /// \param isThisPtr Variable is hidden this pointer
  /// \param forceAsLocal Emit as local even if the variable is a parameter
  /// \param isRefRVal Only relevant for ref/out parameters: indicates whether
  /// ll & addr specify the reference's rvalue, i.e., the lvalue of the original
  /// variable, instead of the reference's lvalue.
  /// \param addr     An array of complex address operations encoding a DWARF
  /// expression.
  void
  EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd, Type *type = nullptr,
                    bool isThisPtr = false, bool forceAsLocal = false,
                    bool isRefRVal = false,
                    llvm::ArrayRef<uint64_t> addr = llvm::ArrayRef<uint64_t>());

  /// \brief Emits all things necessary for making debug info for a global
  /// variable vd.
  /// \param ll       LLVM global variable
  /// \param vd       Variable declaration to emit debug info for.
  void EmitGlobalVariable(llvm::GlobalVariable *ll,
                          VarDeclaration *vd); // FIXME

  void Finalize();

private:
  DIScope GetSymbolScope(Dsymbol *s);
  DIScope GetCurrentScope();
  llvm::StringRef GetNameAndScope(Dsymbol *sym, DIScope &scope);
  void Declare(const Loc &loc, llvm::Value *storage, ldc::DILocalVariable divar,
               ldc::DIExpression diexpr);
  void SetValue(const Loc &loc, llvm::Value *value, ldc::DILocalVariable divar,
                ldc::DIExpression diexpr);
  void AddFields(AggregateDeclaration *sd, ldc::DIFile file,
                 llvm::SmallVector<llvm::Metadata *, 16> &elems);
  void AddStaticMembers(AggregateDeclaration *sd, ldc::DIFile file,
                 llvm::SmallVector<llvm::Metadata *, 16> &elems);
  DIFile CreateFile(const char *filename = nullptr);
  DIFile CreateFile(const Loc &loc);
  DIFile CreateFile(Dsymbol *decl);
  DIType CreateBasicType(Type *type);
  DIType CreateEnumType(TypeEnum *type);
  DIType CreatePointerType(TypePointer *type);
  DIType CreateVectorType(TypeVector *type);
  DIType CreateComplexType(Type *type);
  DIType CreateMemberType(unsigned linnum, Type *type, DIFile file,
                          const char *c_name, unsigned offset, Visibility::Kind,
                          bool isStatic = false, DIScope scope = nullptr);
  DISubprogram CreateFunction(DIScope scope, llvm::StringRef name,
                              llvm::StringRef linkageName, DIFile file,
                              unsigned lineNo, DISubroutineType ty,
                              bool isLocalToUnit, bool isDefinition,
                              bool isOptimized, unsigned scopeLine,
                              DIFlags flags);
  DIType CreateCompositeType(Type *type);
  DIType CreateArrayType(TypeArray *type);
  DIType CreateSArrayType(TypeSArray *type);
  DIType CreateAArrayType(TypeAArray *type);
  DISubroutineType CreateFunctionType(Type *type);
  DISubroutineType CreateEmptyFunctionType();
  DIType CreateDelegateType(TypeDelegate *type);
  DIType CreateUnspecifiedType(Dsymbol *sym);
  DIType CreateTypeDescription(Type *type, bool voidToUbyte = false);

  bool mustEmitFullDebugInfo();
  bool mustEmitLocationsDebugInfo();

  unsigned getColumn(const Loc &loc) const;

public:
  template <typename T>
  void OpOffset(T &addr, llvm::StructType *type, int index) {
    if (!global.params.symdebug) {
      return;
    }

    uint64_t offset =
        gDataLayout->getStructLayout(type)->getElementOffset(index);
    addr.push_back(llvm::dwarf::DW_OP_plus_uconst);
    addr.push_back(offset);
  }

  template <typename T> void OpDeref(T &addr) {
    if (!global.params.symdebug) {
      return;
    }

    addr.push_back(llvm::dwarf::DW_OP_deref);
  }
};

} // namespace ldc
