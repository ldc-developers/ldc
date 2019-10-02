//===-- irstate.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"

#include "gen/mlirstate.h"

#include "llvm/ADT/ScopedHashTable.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

MLIRState *gMLIR = nullptr;

////////////////////////////////////////////////////////////////////////////////
/*This class create a ModuleAST*/



////////////////////////////////////////////////////////////////////////////////
class MLIRGenImpl{
public:
  MLIRGenImpl(mlir::MLIRContext &context) : context(context),
                                            builder(&context) {}

/*
  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    for (FunctionAST &F : moduleAST) {
      auto func = mlirGen(F);
      if (!func)
        return nullptr;
      theModule.push_back(func);
    }

    // FIXME: (in the next chapter...) without registering a dialect in MLIR,
    // this won't do much, but it should at least check some structural
    // properties of the generated MLIR module.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("Module verification error");
      return nullptr;
    }

    return theModule;
  }
*/ /*Create an ModuleAST and a FunctionAST first*/

private:
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// This "module" matches the D source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> symbolTable;



  /// Helper conversion for the DMD AST location to an MLIR location.
  mlir::Location loc(Loc loc){
    llvm::StringRef filename(loc.filename);
    return builder.getFileLineColLoc(builder.getIdentifier(filename), loc.linnum,
                                   loc.charnum);
  }/*Is this a duplicate from dmd/globals.h:404 ? */

  // Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if (this->symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

};
/*namespace Ddialect {
// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace Ddialect*/
