//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DIBUILDER_H
#define LDC_GEN_DIBUILDER_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"

#include "gen/tollvm.h"
#include "mars.h"

struct IRState;

class ClassDeclaration;
class Dsymbol;
class FuncDeclaration;
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
#if LDC_LLVM_VER >= 307
typedef llvm::DIType *DIType;
typedef llvm::DIFile *DIFile;
typedef llvm::DIGlobalVariable *DIGlobalVariable;
typedef llvm::DILocalVariable *DILocalVariable;
typedef llvm::DIExpression *DIExpression;
typedef llvm::DILexicalBlock *DILexicalBlock;
typedef llvm::DIScope *DIScope;
typedef llvm::DISubroutineType *DISubroutineType;
typedef llvm::DISubprogram *DISubprogram;
typedef llvm::DICompileUnit *DICompileUnit;
#else
typedef llvm::DIType DIType;
typedef llvm::DIFile DIFile;
typedef llvm::DIGlobalVariable DIGlobalVariable;
typedef llvm::DIVariable DILocalVariable;
typedef llvm::DILexicalBlock DILexicalBlock;
typedef llvm::DIDescriptor DIScope;
typedef llvm::DICompositeType DISubroutineType;
typedef llvm::DISubprogram DISubprogram;
typedef llvm::DICompileUnit DICompileUnit;
#if LDC_LLVM_VER == 306
typedef llvm::DIExpression DIExpression;
#endif
#endif

class DIBuilder {
  IRState *const IR;
  llvm::DIBuilder DBuilder;

#if LDC_LLVM_VER >= 307
  DICompileUnit CUNode;
#else
  const llvm::MDNode *CUNode;
#endif

  DICompileUnit GetCU() {
#if LDC_LLVM_VER >= 307
    return CUNode;
#else
    return llvm::DICompileUnit(CUNode);
#endif
  }

public:
  explicit DIBuilder(IRState *const IR);

  /// \brief Emit the Dwarf compile_unit global for a Module m.
  /// \param m        Module to emit as compile unit.
  void EmitCompileUnit(Module *m);

  /// \brief Emit the Dwarf subprogram global for a function declaration fd.
  /// \param fd       Function declaration to emit as subprogram.
  /// \returns        the Dwarf subprogram global.
  DISubprogram EmitSubProgram(FuncDeclaration *fd); // FIXME

  /// \brief Emit the Dwarf subprogram global for a thunk.
  /// \param Thunk    llvm::Function pointer.
  /// \param fd       The function wrapped by this thunk.
  /// \returns        the Dwarf subprogram global.
  DISubprogram EmitThunk(llvm::Function *Thunk,
                              FuncDeclaration *fd); // FIXME

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

  /// \brief Emits debug info for function end
  void EmitFuncEnd(FuncDeclaration *fd);

  /// \brief Emits debug info for block start
  void EmitBlockStart(Loc &loc);

  /// \brief Emits debug info for block end
  void EmitBlockEnd();

  void EmitStopPoint(Loc &loc);

  void EmitValue(llvm::Value *val, VarDeclaration *vd);

  /// \brief Emits all things necessary for making debug info for a local
  /// variable vd.
  /// \param ll       LLVM Value of the variable.
  /// \param vd       Variable declaration to emit debug info for.
  /// \param type     Type of parameter if diferent from vd->type
  /// \param isThisPtr Parameter is hidden this pointer
  /// \param fromNested Is a closure variable accessed through nest_arg
  /// \param addr     An array of complex address operations.
  void
  EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd, Type *type = nullptr,
                    bool isThisPtr = false, bool fromNested = false,
#if LDC_LLVM_VER >= 306
                    llvm::ArrayRef<int64_t> addr = llvm::ArrayRef<int64_t>()
#else
                    llvm::ArrayRef<llvm::Value *> addr =
                        llvm::ArrayRef<llvm::Value *>()
#endif
                        );

  /// \brief Emits all things necessary for making debug info for a global
  /// variable vd.
  /// \param ll       LLVM global variable
  /// \param vd       Variable declaration to emit debug info for.
  DIGlobalVariable EmitGlobalVariable(llvm::GlobalVariable *ll,
                                      VarDeclaration *vd); // FIXME

  void Finalize();

private:
  llvm::LLVMContext &getContext();
  Module *getDefinedModule(Dsymbol *s);
  DIScope GetCurrentScope();
  void Declare(const Loc &loc, llvm::Value *var, ldc::DILocalVariable divar
#if LDC_LLVM_VER >= 306
               ,
               ldc::DIExpression diexpr
#endif
               );
  void AddBaseFields(ClassDeclaration *sd, ldc::DIFile file,
#if LDC_LLVM_VER >= 306
                     std::vector<llvm::Metadata *> &elems
#else
                     std::vector<llvm::Value *> &elems
#endif
                     );
  DIFile CreateFile(Loc &loc);
  DIType CreateBasicType(Type *type);
  DIType CreateEnumType(Type *type);
  DIType CreatePointerType(Type *type);
  DIType CreateVectorType(Type *type);
  DIType CreateMemberType(unsigned linnum, Type *type, DIFile file,
                          const char *c_name, unsigned offset, PROTKIND);
  DIType CreateCompositeType(Type *type);
  DIType CreateArrayType(Type *type);
  DIType CreateSArrayType(Type *type);
  DIType CreateAArrayType(Type *type);
  DISubroutineType CreateFunctionType(Type *type);
  DISubroutineType CreateDelegateType(Type *type);
  DIType CreateTypeDescription(Type *type, bool derefclass = false);

public:
  template <typename T>
  void OpOffset(T &addr, llvm::StructType *type, int index) {
    if (!global.params.symdebug) {
      return;
    }

    uint64_t offset =
        gDataLayout->getStructLayout(type)->getElementOffset(index);
#if LDC_LLVM_VER >= 306
    addr.push_back(llvm::dwarf::DW_OP_plus);
    addr.push_back(offset);
#else
    llvm::Type *int64Ty = llvm::Type::getInt64Ty(getContext());
    addr.push_back(llvm::ConstantInt::get(int64Ty, llvm::DIBuilder::OpPlus));
    addr.push_back(llvm::ConstantInt::get(int64Ty, offset));
#endif
  }

  template <typename T> void OpOffset(T &addr, llvm::Value *val, int index) {
    if (!global.params.symdebug) {
      return;
    }

    llvm::StructType *type = isaStruct(val->getType()->getContainedType(0));
    assert(type);
    OpOffset(addr, type, index);
  }

  template <typename T> void OpDeref(T &addr) {
    if (!global.params.symdebug) {
      return;
    }

#if LDC_LLVM_VER >= 306
    addr.push_back(llvm::dwarf::DW_OP_deref);
#else
    llvm::Type *int64Ty = llvm::Type::getInt64Ty(getContext());
    addr.push_back(llvm::ConstantInt::get(int64Ty, llvm::DIBuilder::OpDeref));
#endif
  }
};

} // namespace ldc

#endif
