//
// Created by Roberto Rosmaninho on 09/10/19.
//

#ifndef LDC_MLIRHELLPERS_H
#define LDC_MLIRHELLPERS_H

#include "dmd/declaration.h"
#include "dmd/statement.h"
#include "dmd/mtype.h"
#include <vector>
#include "dmd/compiler.h"
#include "dmd/dsymbol.h"
#include "gen/logger.h"
#include "mlir/IR/Value.h"

//std::vector<Statement> vec_stm;
//std::vector<Expression> vec_exp;
class MLIRState {
private:
  std::vector<FuncDeclaration*> vec_functions;

public:
  MLIRState(){}
  ~MLIRState(){}
  std::vector<FuncDeclaration*> getFunctions(){
    return vec_functions;
  }
};

mlir::Value *DtoDeclarationExpMLIR(Dsymbol *declaration);

void DtoResolveDsymbolMLIR(Dsymbol *dsym);
void Declaration_MLIRcodegen(Dsymbol *decl);

#endif // LDC_MLIRHELLPERS_H
