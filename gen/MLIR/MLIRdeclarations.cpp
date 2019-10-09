//
// Created by Roberto Rosmaninho on 09/10/19.
//
#include "gen/MLIR/MLIRhelpers.h"
#include "gen/logger.h"
#include "dmd/dsymbol.h"
#include "dmd/visitor.h"
#include "dmd/import.h"
#include <string>

class MLIRCodegenVisitor : public Visitor {
  MLIRState *mlir_;
public:
  explicit MLIRCodegenVisitor(MLIRState *mlir){
    this->mlir_ = mlir;
  }

  //////////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////////

  void visit(Dsymbol *sym) override {
    IF_LOG Logger::println("Ignoring Dsymbol::codegen for %s",
                           sym->toPrettyChars());
  }

  void visit(Import *im) override {
    //Deleting filename from module: filename.import
    std::string s = im->toPrettyChars();
    s.erase(0, s.rfind(".") + 1);
    if(s.compare("object") != 0) {
      // Do something
      IF_LOG Logger::println("Import::mlircodegen for %s", im->toPrettyChars());
      // irs->DBuilder.EmitImport(im);
    }else{
      IF_LOG Logger::println("Ignoring import %s", im->toPrettyChars());
    }
    LOG_SCOPE
  }

  //////////////////////////////////////////////////////////////////////////////

 // void visit(VarDeclaration *vdecl) override {

 // }

  //////////////////////////////////////////////////////////////////////////

  void visit(FuncDeclaration *decl) override {
    // don't touch function aliases, they don't contribute any new symbols
    if (!decl->isFuncAliasDeclaration()) {
     mlir_->getFunctions().push_back(decl);
    }
  }
};
  //////////////////////////////////////////////////////////////////////////////
  void Declaration_MLIRcodegen(Dsymbol *decl){
      IF_LOG Logger::println("MLIRCodeGen Dsymbol: '%s'", decl->toPrettyChars());
      MLIRState *mlir_ = new MLIRState();
    MLIRCodegenVisitor vis(mlir_);
    decl->accept(&vis);
  }

