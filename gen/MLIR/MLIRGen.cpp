//
// Created by Roberto Rosmaninho on 07/10/19.
//

#include <dmd/declaration.h>
#include <dmd/statement.h>
#include <dmd/declaration.h>
#include "dmd/globals.h"
#include "dmd/module.h"

#include "gen/llvmhelpers.h"
#include "ir/irdsymbol.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "gen/MLIR/MLIRhelpers.h"

#include <memory>
#include "MLIRGen.h"

using namespace ldc_mlir;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context)
              : context(context), builder(&context) {}


   mlir::ModuleOp mlirGen(Module *m){
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    for(unsigned k = 0; k < m->members->dim; k++){
      Dsymbol *dsym = (*m->members)[k];
      assert(dsym);

      Declaration_MLIRcodegen(dsym);
      FuncDeclaration *fd = dsym->isFuncDeclaration();
      if(fd != NULL) {
        auto func = mlirGen(fd);
        if (!func)
          return nullptr;
        theModule.push_back(func);
      }
    }

     // this won't do much, but it should at least check some structural
     // properties of the generated MLIR module.
   //  if (failed(mlir::verify(theModule))) {
    //   theModule.emitError("module verification error");
     //  return nullptr;
    // }

     return theModule;

  } //MLIRCodeIMplementatio for a given Module

private:
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// A "module" matches a Toy source file: containing a list of functions.
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

  mlir::Location loc(Loc loc){
    return builder.getFileLineColLoc(builder.getIdentifier(StringRef(loc
    .filename)),
        loc.linnum, loc.charnum);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if(symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  //Create the DSymbol for an MLIR Function with as many argument as the
  //provided by Module
  mlir::FuncOp mlirGen(FuncDeclaration *Fd, bool level){
    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> ret_types;
    // Arguments type is uniformly a generic array.
    VarDeclaration *vd = Fd->v_arguments;
    auto type = builder.getIntegerType(16);

    llvm::SmallVector<mlir::Type, 4> arg_types;//(1,type);

    auto func_type = builder.getFunctionType(arg_types, ret_types);
    auto function = mlir::FuncOp::create(loc(Fd->loc),
        StringRef(Fd->toPrettyChars()),
        func_type, {});

    // Mark the function as generic: it'll require type specialization for every
    // call site.
    if (function.getNumArguments()) {
      function.setAttr("ldc_mlir.generic", builder.getUnitAttr());
    }

    return function;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FuncDeclaration *Fd) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen(Fd, 1));
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto &protoArgs = Fd->parameters;

    // Declare all the function arguments in the symbol table.
 /*   for (const auto &name_value :
        llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value)->getName(),
                         std::get<1>(name_value))))
        return nullptr;
    }
*/
    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
   builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
   // if (mlir::failed(mlirGen(Fd->fbody))) {
   //   function.erase();
   //   return nullptr;
   // }

    // Implicitly return void if no return statement was emitted.
    // (this would possibly help the REPL case later)
   // if (function.getBody().back().back().getName().getStringRef() !=
   //     "ldc_mlir.return") {
      //ReturnStatement *rS;
      //mlirGen(rS);
   // }

    return function;

  }
}; //class MLIRGenImpl
} //annonymous namespace

namespace ldc_mlir {
// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              Module *m) {
  return MLIRGenImpl(context).mlirGen(m);
}
} //ldc_mlir namespce