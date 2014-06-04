//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC � the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DIBUILDER_H
#define LDC_GEN_DIBUILDER_H

#if LDC_LLVM_VER >= 303
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DataLayout.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#else
#include "llvm/DebugInfo.h"
#include "llvm/DIBuilder.h"
#endif
#else
#if LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#include "llvm/DebugInfo.h"
#include "llvm/DIBuilder.h"
#else
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Analysis/DebugInfo.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/DIBuilder.h"
#else
#include "llvm/Analysis/DIBuilder.h"
#endif
#include "llvm/Target/TargetData.h"
#endif
#endif

#include "gen/tollvm.h"
#include "mars.h"

struct IRState;

struct ClassDeclaration;
struct Dsymbol;
struct FuncDeclaration;
struct Module;
struct Type;
struct VarDeclaration;

namespace llvm {
    class GlobalVariable;
    class StructType;
    class LLVMContext;

// Only for the OpXXX templates, see below.
#if LDC_LLVM_VER >= 302
    class DataLayout;
#else
    class TargetData;
#endif
}

// Only for the OpXXX templates, see below.
#if LDC_LLVM_VER >= 302
extern const llvm::DataLayout* gDataLayout;
#else
extern const llvm::TargetData* gDataLayout;
#endif

namespace ldc {

#if LDC_LLVM_VER >= 304
typedef llvm::DICompositeType DIFunctionType;
#else
typedef llvm::DIType DIFunctionType;
#endif

class DIBuilder
{
    IRState *const IR;
    const llvm::MDNode *CUNode;
#if LDC_LLVM_VER >= 305
    llvm::DIBuilder DBuilder;
#else
    llvm::DIBuilder DBuilder;
#endif

    const llvm::MDNode *GetCU()
    {
        return CUNode;
    }

public:
    DIBuilder(IRState *const IR, llvm::Module &M);

    /// \brief Emit the Dwarf compile_unit global for a Module m.
    /// \param m        Module to emit as compile unit.
    void EmitCompileUnit(Module *m);

    /// \brief Emit the Dwarf subprogram global for a function declaration fd.
    /// \param fd       Function declaration to emit as subprogram.
    /// \returns        the Dwarf subprogram global.
    llvm::DISubprogram EmitSubProgram(FuncDeclaration *fd); // FIXME

    /// \brief Emit the Dwarf subprogram global for a internal function.
    /// This is used for generated functions like moduleinfoctors,
    /// module ctors/dtors and unittests.
    /// \param prettyname   The name as seen in the source.
    /// \param mangledname  The mangled name in the object file.
    /// \returns       the Dwarf subprogram global.
    llvm::DISubprogram EmitSubProgramInternal(llvm::StringRef prettyname, llvm::StringRef mangledname);  // FIXME

    /// \brief Emits debug info for function start
    void EmitFuncStart(FuncDeclaration *fd);

    /// \brief Emits debug info for function end
    void EmitFuncEnd(FuncDeclaration *fd);

    /// \brief Emits debug info for block start
    void EmitBlockStart(Loc loc);

    /// \brief Emits debug info for block end
    void EmitBlockEnd();

    void EmitStopPoint(unsigned ln);

    void EmitValue(llvm::Value *val, VarDeclaration* vd);

    /// \brief Emits all things necessary for making debug info for a local variable vd.
    /// \param ll       LLVM Value of the variable.
    /// \param vd       Variable declaration to emit debug info for.
    /// \param addr     An array of complex address operations.
    void EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                           llvm::ArrayRef<llvm::Value *> addr = llvm::ArrayRef<llvm::Value *>());

    /// \brief Emits all things necessary for making debug info for a global variable vd.
    /// \param ll       LLVM global variable
    /// \param vd       Variable declaration to emit debug info for.
    llvm::DIGlobalVariable EmitGlobalVariable(llvm::GlobalVariable *ll, VarDeclaration *vd); // FIXME

    void EmitModuleEnd();

private:
    llvm::LLVMContext &getContext();
    Module *getDefinedModule(Dsymbol *s);
    llvm::DIDescriptor GetCurrentScope();
    void Declare(llvm::Value *var, llvm::DIVariable divar);
    void AddBaseFields(ClassDeclaration *sd, llvm::DIFile file,
                         std::vector<llvm::Value*> &elems);
    llvm::DIFile CreateFile(Loc loc);
    llvm::DIType CreateBasicType(Type *type);
    llvm::DIType CreateEnumType(Type *type);
    llvm::DIType CreatePointerType(Type *type);
    llvm::DIType CreateVectorType(Type *type);
    llvm::DIType CreateMemberType(unsigned linnum, Type *type, llvm::DIFile file, const char* c_name, unsigned offset);
    llvm::DIType CreateCompositeType(Type *type);
    llvm::DIType CreateArrayType(Type *type);
    llvm::DIType CreateSArrayType(Type *type);
    llvm::DIType CreateAArrayType(Type *type);
    DIFunctionType CreateFunctionType(Type *type);
    DIFunctionType CreateDelegateType(Type *type);
    llvm::DIType CreateTypeDescription(Type* type, const char* c_name, bool derefclass = false);

public:
    template<typename T>
    void OpOffset(T &addr, llvm::StructType *type, int index)
    {
        if (!global.params.symdebug)
            return;

        uint64_t offset = gDataLayout->getStructLayout(type)->getElementOffset(index);
        llvm::Type *int64Ty = llvm::Type::getInt64Ty(getContext());
        addr.push_back(llvm::ConstantInt::get(int64Ty, llvm::DIBuilder::OpPlus));
        addr.push_back(llvm::ConstantInt::get(int64Ty, offset));
    }

    template<typename T>
    void OpOffset(T &addr, llvm::Value *val, int index)
    {
        if (!global.params.symdebug)
            return;

        llvm::StructType *type = isaStruct(val->getType()->getContainedType(0));
        assert(type);
        OpOffset(addr, type, index);
    }

    template<typename T>
    void OpDeref(T &addr)
    {
        if (!global.params.symdebug)
            return;

        llvm::Type *int64Ty = llvm::Type::getInt64Ty(getContext());
        addr.push_back(llvm::ConstantInt::get(int64Ty, llvm::DIBuilder::OpDeref));
    }
};


} // namespace ldc

#endif
