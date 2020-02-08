//
// Created by Roberto Rosmaninho on 09/10/19.
//

#ifndef LDC_TOMLIR_H
#define LDC_TOMLIR_H

#include "dmd/module.h"
#include "mlir/IR/MLIRContext.h"

void writeMLIRModule(Module *m, mlir::MLIRContext &mlirContext,
                                                  const char *filename);

#endif // LDC_TOMLIR_H
