//===-- MLIRGen.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/globals.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/import.h"
#include "dmd/module.h"
#include "dmd/statement.h"

#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/MLIR/MLIRGen.h"
#include "gen/MLIR/MLIRStatements.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <memory>

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
  MLIRGenImpl(mlir::MLIRContext &context, IRState *irs)
      : irs(irs), context(context), builder(&context) {}

  mlir::ModuleOp mlirGen(Module *m){
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    m->ir->resetAll();

    for(unsigned long k = 0; k < m->members->dim; k++) {
      total++;
      Dsymbol *dsym = (*m->members)[k];
      assert(dsym);

      MLIRDeclaration *declaration = new MLIRDeclaration(irs, m, context,
          builder,symbolTable,total,miss);

      // Declaration_MLIRcodegen(dsym, mlir_);
      FuncDeclaration *fd = dsym->isFuncDeclaration();
      if (fd != nullptr) {
        auto func = mlirGen(fd);
        if (!func)
          return nullptr;
        theModule.push_back(func);
      } else if (dsym->isInstantiated()) {
        IF_LOG Logger::println("isTemplateInstance: '%s'",
                               dsym->isTemplateInstance()->toChars());
      } else if (dsym->isImport()) {
        IF_LOG Logger::println("isImport: %s", dsym->isImport()->toChars());
      } else if (dsym->isVarDeclaration()) {
        IF_LOG Logger::println("isVarDeclaration: '%s'",
                               dsym->isVarDeclaration()->toChars());
      } else if (ScopeDsymbol *scopeDsymbol = dsym->isScopeDsymbol()) {
        IF_LOG Logger::println("isScopeDsymbol: '%s'", scopeDsymbol->toChars());
        LOG_SCOPE

        if(auto *templateInstance = scopeDsymbol->isTemplateInstance()) {
          declaration->mlirGen(templateInstance);
        }
      }else{
        IF_LOG Logger::println("Unnable to recoganize dsym member: '%s'",
            dsym->toPrettyChars());
        miss++;
      }
    }

    // this won't do much, but it should at least check some structural
    // properties of the generated MLIR module.
      if (failed(mlir::verify(theModule))) {
       theModule.emitError("module verification error");
       total++;
       miss++;
      return nullptr;
     }

      IF_LOG Logger::println("#### Total: '%u'", total);
      IF_LOG Logger::println("### Miss: '%u'", miss);

    return theModule;

  } //MLIRCodeImplementation for a given Module

private:

  /// Getting IRState to have access to all Statments and Declarations of
  // programs
  IRState *irs;

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


  /// This flags counts the number of hits and misses of our translation.
  unsigned total = 0, miss = 0;

  mlir::Location loc(Loc loc){
    return builder.getFileLineColLoc(builder.getIdentifier(
        StringRef(loc.filename)),loc.linnum, loc.charnum);
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

    //Supposing that the type is integer
    auto type = builder.getIntegerType(32);
    unsigned long size = 0;
    if(Fd->parameters)
      size = Fd->parameters->dim;

    // Arguments type is uniformly a generic array.
    llvm::SmallVector<mlir::Type, 4> arg_types(size, type);

    auto func_type = builder.getFunctionType(arg_types, ret_types);
    auto function = mlir::FuncOp::create(loc(Fd->loc),
                                         StringRef(Fd->mangleString),
                                         func_type, {});

    // Mark the function as generic: it'll require type specialization for every
    // call site.
    if (function.getNumArguments()) {
      function.setAttr("ldc.generic", builder.getUnitAttr());
    }

    return function;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FuncDeclaration *Fd) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function = mlirGen(Fd, true);
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Initialize the object to be the "visitor"
    MLIRStatements *genStmt = new MLIRStatements(irs, irs->dmodule, context,
                                                 builder, symbolTable, total,
                                                 miss);

    //Setting arguments of a given function
    unsigned long size = 0;
    if(Fd->parameters)
      size = Fd->parameters->dim;
    llvm::SmallVector<VarDeclarations*, 4> args(size, Fd->parameters);

    //args.push_back(mlirGen())
    auto &protoArgs = args;

    // Declare all the function arguments in the symbol table.
    for (auto name_value : llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value)->pop()->toChars(),
                         std::get<1>(name_value))))
        return nullptr;
    }
    // Emit the body of the function.

    mlir::LogicalResult result = genStmt->genStatements(Fd);
    if (mlir::failed(result)) {
      function.erase();
      return nullptr;
    }
    //function.getBody().back().back().getParentRegion()->viewGraph();

    // Implicitly return void if no return statement was emitted.
    // (this would possibly help the REPL case later)
    if (function.getBody().back().back().getName().getStringRef() !=
        "ldc.return") {
      ReturnStatement *returnStatement = Fd->returns->front();
      if(returnStatement != nullptr)
        genStmt->mlirGen(returnStatement);
      else{
        mlir::ReturnOp returnOp;
        mlir::UnknownLoc unknownLoc;
        mlir::OperationState ret_result(unknownLoc, "ldc.return");
        returnOp.build(&builder, ret_result);
      }
    }
    return function;
  }


}; //class MLIRGenImpl
} //annonymous namespace

namespace ldc_mlir {
// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              Module *m, IRState *irs) {
  return MLIRGenImpl(context, irs).mlirGen(m);
}
} //ldc_mlir namespce

#endif //LDC_MLIR_ENABLED
