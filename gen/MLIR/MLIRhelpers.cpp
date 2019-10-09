//
// Created by Roberto Rosmaninho on 09/10/19.
//

#include <mlir/IR/Value.h>
#include "gen/MLIR/MLIRhelpers.h"



void DtoResolveDsymbolMLIR(Dsymbol *dsym){
  IF_LOG Logger::println("");
}

mlir::Value *DtoDeclarationExpMLIR(Dsymbol *declaration){
  IF_LOG Logger::print("DtoDeclarationExpMLIR: %s\n", declaration->toChars());
  LOG_SCOPE;

  if(VarDeclaration *vd = declaration->isVarDeclaration()){
    Logger::println("VarDeclaration");

    // if aliassym is set, this VarDecl is redone as an alias to another symbol
    // this seems to be done to rewrite Tuple!(...) v;
    // as a TupleDecl that contains a bunch of individual VarDecls
    if (vd->aliassym)
      return DtoDeclarationExpMLIR(vd->aliassym);


  if(vd->isDataseg())
    Declaration_MLIRcodegen(vd);
 // else
   // DtoVarDeclaration(vd);
  }
  return nullptr; //temporary

  /*
  if (StructDeclaration *s = declaration->isStructDeclaration()) {
    Logger::println("StructDeclaration");
   // Declaration_codegen(s);
  } else if (FuncDeclaration *f = declaration->isFuncDeclaration()) {
    Logger::println("FuncDeclaration");
    Declaration_MLIRcodegen(f);
  } else if (ClassDeclaration *e = declaration->isClassDeclaration()) {
    Logger::println("ClassDeclaration");
    Declaration_codegen(e);
  } else if (AttribDeclaration *a = declaration->isAttribDeclaration()) {
    Logger::println("AttribDeclaration");
    // choose the right set in case this is a conditional declaration
    if (auto d = a->include(nullptr)) {
      for (unsigned i = 0; i < d->dim; ++i) {
        DtoDeclarationExp((*d)[i]);
      }
    }
  } else if (TemplateMixin *m = declaration->isTemplateMixin()) {
    Logger::println("TemplateMixin");
    for (Dsymbol *mdsym : *m->members) {
      DtoDeclarationExp(mdsym);
    }
  } else if (TupleDeclaration *tupled = declaration->isTupleDeclaration()) {
    Logger::println("TupleDeclaration");
    assert(tupled->isexp && "Non-expression tuple decls not handled yet.");
    assert(tupled->objects);
    for (unsigned i = 0; i < tupled->objects->dim; ++i) {
      auto exp = static_cast<DsymbolExp *>((*tupled->objects)[i]);
      DtoDeclarationExp(exp->s);
    }
  } else {
    // Do nothing for template/alias/enum declarations and static
    // assertions. We cannot detect StaticAssert without RTTI, so don't
    // even bother to check.
    IF_LOG Logger::println("Ignoring Symbol: %s", declaration->kind());
  }*/
}