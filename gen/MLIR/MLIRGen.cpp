//
// Created by Roberto Rosmaninho on 07/10/19.
//

#include <dmd/declaration.h>
#include <dmd/statement.h>
#include <dmd/declaration.h>
#include "dmd/globals.h"
#include "dmd/module.h"
#include "dmd/expression.h"
#include "dmd/statement.h"
#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/init.h"

#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/logger.h"
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
#include "dmd/declaration.h"
#include "dmd/identifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"


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
  MLIRGenImpl(mlir::MLIRContext &context, IRState *irs)
      : context(context), irs(irs), builder(&context) {}

 // MLIRState *mlir_ = new MLIRState();

  mlir::ModuleOp mlirGen(Module *m){
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    for(unsigned long k = 0; k < m->members->dim; k++){
      Dsymbol *dsym = (*m->members)[k];
      assert(dsym);

      //Declaration_MLIRcodegen(dsym, mlir_);
      FuncDeclaration *fd = dsym->isFuncDeclaration();
      if(fd != nullptr) {
        auto func = mlirGen(fd);
        if (!func)
          return nullptr;
        theModule.push_back(func);
      }
    }

    // this won't do much, but it should at least check some structural
    // properties of the generated MLIR module.
      if (failed(mlir::verify(theModule))) {
       theModule.emitError("module verification error");
      return nullptr;
     }

    return theModule;

  } //MLIRCodeIMplementatio for a given Module

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
    auto type = builder.getIntegerType(16);
    unsigned long size = 0;
    if(Fd->parameters)
      size = Fd->parameters->dim;

    llvm::SmallVector<mlir::Type, 4> arg_types(size, type);

    auto func_type = builder.getFunctionType(arg_types, ret_types);
    auto function = mlir::FuncOp::create(loc(Fd->loc),
                                         StringRef(Fd->toPrettyChars()), func_type, {});

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
    mlir::FuncOp function = mlirGen(Fd, true);
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
   //mlir::Value **value = nullptr;
   // unsigned i = 0;
   //for(auto arg : *Fd->parameters)
   //  mlirGen(arg);
   //   value[i++] = mlirGen(arg);
   // }
    //auto &protoArgs = reinterpret_cast<const mlir::BlockArgument &>(value);

    // Declare all the function arguments in the symbol table.
   /* for (auto name_value :// TODO: Translate Parmeters to Arguments
        llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value)->getName(),
                         std::get<1>(name_value))))
        return nullptr;
    }*/

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(Fd->fbody))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // (this would possibly help the REPL case later)
    if (function.getBody().back().back().getName().getStringRef() !=
        "ldc.return") {
      ReturnStatement *returnStatement = Fd->returns->front();
      mlirGen(returnStatement);
    }
    return function;
  }

  mlir::LogicalResult mlirGen(ReturnStatement *returnStatement){
    if(!returnStatement->exp->isIntegerExp())
      return mlir::failure();

    IF_LOG Logger::println("MLIRCodeGen - Return Stmt: '%s'",
        returnStatement->toChars());
    LOG_SCOPE

    mlir::OperationState result(loc(returnStatement->loc),"ldc.return");
    if(returnStatement->exp->hasCode()) {
      auto *expr = mlirGen(returnStatement->exp);
      if(!expr)
        return mlir::failure();
      result.addOperands(expr);
    }
    builder.createOperation(result);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(Statement *stmt){
    auto vec_stm = irs->getExpStatements();
    for(auto stm : vec_stm) {
      if (CompoundStatement *compoundStatement = stm->isCompoundStatement()) {
        IF_LOG Logger::println("MLIRCodeGen - CompundStatement: '%s'",
                               compoundStatement->toChars());
        LOG_SCOPE
        mlirGen(compoundStatement);
        //return mlir::success();
      }
      if(ExpStatement *expStatement = stm->isExpStatement()){
        IF_LOG Logger::println("MLIRCodeGen ExpStatement");
        LOG_SCOPE
        mlirGen(expStatement);
      }else {
        IF_LOG Logger::println("Unable to recoganize Statament: '%s'",
                               stm->toChars());
        mlir::failure();
      }
    }
    auto vec_ret = irs->getReturnStatements();//getReturnStatements();
    for(auto ret : vec_ret)
      mlirGen(ret);

    return mlir::success();
  }

  mlir::Value *mlirGen(CompoundStatement *cstm){
    for(auto stmt : *cstm->statements){
      if(ExpStatement *expStmt = stmt->isExpStatement()) {
        IF_LOG Logger::println("MLIRCodeGen ExpStatement");
        LOG_SCOPE
        return mlirGen(expStmt);
      }else{
        IF_LOG Logger::println("Unable to recoganize: '%s'", stmt->toChars());
      }
    }
    return nullptr;
  }

  mlir::Value *mlirGen(ExpStatement *expStmt){
    IF_LOG Logger::println("MLIRCodeGen: ExpStatement to MLIR: '%s'",
                           expStmt->toChars());
    LOG_SCOPE
    auto location = loc(expStmt->loc);
    mlir::Value *value = nullptr;

    if(DeclarationExp *decl_exp = expStmt->exp->isDeclarationExp()){
      value = mlirGen(decl_exp);
    }else if(Expression *e = expStmt->exp) {
      IF_LOG Logger::println("Passou Aqui");
      //toMLIRElem(e, mlir_);
      if(DeclarationExp *edecl = e->isDeclarationExp()){
        IF_LOG Logger::println("Declaration");
      }
    }else{
      IF_LOG Logger::println("Unable to recoganize: '%s'",
          expStmt->exp->toChars());
      return nullptr;
    }
    mlir::OperationState result(location, "ldc.ExpStatment");
    result.addTypes(builder.getIntegerType(16));
    result.addOperands(value);
    return builder.createOperation(result)->getResult(0);
  }

  mlir::Value *mlirGen(Expression *exp){
    IF_LOG Logger::println("MLIRCodeGen - Expression: '%s'",
        exp->toChars
    ());
    LOG_SCOPE

    const char *op_name = nullptr;
    auto location = loc(exp->loc);
    mlir::Value *e1 = nullptr;
    mlir::Value *e2 = nullptr;

    if(AssignExp *assign = exp->isAssignExp()){
      IF_LOG Logger::print("AssignExp::toElem: %s | (%s)(%s = %s)\n",
                           assign->toChars(), assign->type->toChars(),
                           assign->e1->type->toChars(),
                           assign->e2->type ? assign->e2->type->toChars() :nullptr);
      if(assign->memset & referenceInit){
        IF_LOG Logger::println(" --> Expression op: '%d'", assign->e1->op);
        assert(assign->op == TOKconstruct || assign->op == TOKblit);
        assert(assign->e1->op == TOKvar);

        Declaration *d = static_cast<VarExp *>(assign->e1)->var;
        if(d->storage_class & (STCref | STCout)) {
          mlir::Value *value = mlirGen(assign->e2);

          if (!value)
            return nullptr;

          mlir::OperationState result(loc(exp->loc), "assign");
          result.addTypes(builder.getIntegerType(16)); // TODO: type
          result.addOperands(value);
          value = builder.createOperation(result)->getResult(0);

          if (failed(declare(assign->e1->toChars(), value)))
            return nullptr;
          return value;
        }
      }
    }else if(ConstructExp *constructExp = exp->isConstructExp()) {
      IF_LOG Logger::println("MLIRCodeGen - ConstructExp");
      LOG_SCOPE
//      mlir::Value *lhs = mlirGen(constructExp->e1);
      mlir::Value *rhs = mlirGen(constructExp->e2);
      mlir::Value *value = nullptr;

      mlir::OperationState result(loc(exp->loc), "ldc.ConstructExp");
      result.addTypes(builder.getIntegerType(16)); // TODO: type
      result.addOperands(rhs);
      value = builder.createOperation(result)->getResult(0);

      if (failed(declare(constructExp->e1->toChars(), rhs)))
        return nullptr;
      return value;
    }else

    if(exp->isDeclarationExp())
      Logger::println("Declaration");
    else if(exp->isAssertExp())
      Logger::println("Assert");
    else if(exp->isDefaultInitExp())
      Logger::println("DefaultInitExp");
    else if(exp->isDsymbolExp())
      Logger::println("Dsymbol");
    else if(exp->isIntegerExp()) {
      dinteger_t dint = exp->isIntegerExp()->value;
      Logger::println("Integer: '%lu'", dint);
      mlir::OperationState result(loc(exp->loc), "ldc.IntegerExp");
      result.addTypes(builder.getIntegerType(16)); // TODO: type
      result.addAttribute("value", builder.getI16IntegerAttr(dint));
      return builder.createOperation(result)->getResult(0);
    }
    else if(VarExp *varExp = exp->isVarExp()) {
      Logger::println("VarExp: '%s'", varExp->var->toChars());
      auto var = symbolTable.lookup(varExp->var->toChars());
      if(!var){
        IF_LOG Logger::println("Undeclared VarExp: '%s'", varExp->toChars());
        return nullptr;
      }else{
        return var;
      }

    }else if(AddExp *add = exp->isAddExp()) {
      e1 = mlirGen(add->e1);
      e2 =mlirGen(add->e2);
      op_name = "ldc.add";
    }else if(MinExp *min = exp->isMinExp()) {
      e1 = mlirGen(min->e1);
      e2 = mlirGen(min->e2);
      op_name = "ldc.neg";
    }else if(MulExp *mul = exp->isMulExp()) {
      e1 = mlirGen(mul->e1);
      e2 = mlirGen(mul->e2);
      op_name = "ldc.mul";
    }else if(DivExp *div = exp->isDivExp()) {
      e1 = mlirGen(div->e1);
      e2 = mlirGen(div->e2);
      op_name = "ldc.div";
    }

    if(e1 != nullptr && e2 != nullptr){
      mlir::OperationState result(location, op_name);
      result.addTypes(builder.getIntegerType(16)); //TODO: MODIFY TO ALLOW MORE TYPES
      result.addOperands({e1, e2});
      return builder.createOperation(result)->getResult(0);
    }else{
      Logger::println("Unable to recoganize the Expression: '%s'",
          exp->toChars());
      return nullptr;
    }
  }

  mlir::Value *mlirGen(VarDeclaration *vd){
    IF_LOG Logger::println("MLIRCodeGen - VarDeclaration: '%s'", vd->toChars
          ());
    LOG_SCOPE
    // if aliassym is set, this VarDecl is redone as an alias to another symbol
    // this seems to be done to rewrite Tuple!(...) v;
    // as a TupleDecl that contains a bunch of individual VarDecls
    if (vd->aliassym) {
      IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: aliassym -> "
                             "APAGAR");
      //return DtoDeclarationExpMLIR(vd->aliassym, mlir_);
    }
    if (vd->isDataseg()) {
      IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: dataseg -> "
                             "APAGAR");
      //Declaration_MLIRcodegen(vd, mlir_);
    }else {
      if (vd->nestedrefs.dim) {
        IF_LOG Logger::println(
              "has nestedref set (referenced by nested function/delegate)");

        // A variable may not be really nested even if nextedrefs is not empty
        // in case it is referenced by a function inside __traits(compile) or
        // typeof.
        // assert(vd->ir->irLocal && "irLocal is expected to be already set by
        // DtoCreateNestedContext");
      }

      if (vd->_init) {
        if (ExpInitializer *ex = vd->_init->isExpInitializer()) {
          // TODO: Refactor this so that it doesn't look like toElem has no effect.
          Logger::println("MLIRCodeGen - ExpInitializer: '%s'", ex->toChars());
          LOG_SCOPE
          return mlirGen(ex->exp);
        }
      }else{
        IF_LOG Logger::println("Unable to recoganize VarDeclaration: '%s'",
                               vd->toChars());
      }
    }
    return nullptr;
  }


  mlir::Value *mlirGen(DeclarationExp *decl_exp){
    IF_LOG Logger::println("MLIRCodeGen - DeclExp: '%s'", decl_exp->toChars());
    LOG_SCOPE
    Dsymbol *dsym = decl_exp->declaration;


    if (VarDeclaration *vd = dsym->isVarDeclaration())
      return mlirGen(vd);

    IF_LOG Logger::println("Unable to recoganize DeclarationExp: '%s'",
                           dsym->toChars());
    return nullptr;
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
