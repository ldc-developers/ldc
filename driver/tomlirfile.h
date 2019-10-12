//
// Created by Roberto Rosmaninho on 09/10/19.
//

#ifndef LDC_TOMLIR_H
#define LDC_TOMLIR_H

#include "gen/irstate.h"
#include "dmd/module.h"
#include "mlir/IR/MLIRContext.h"

void writeMLIRModule(Module *m, mlir::MLIRContext &mlirContext,
    const char *filename, IRState *irs);
#endif // LDC_TOMLIR_H
