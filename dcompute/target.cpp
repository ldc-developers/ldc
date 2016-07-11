//===-- targetCUDA.cpp ------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/target.h"
#include "dsymbol.h"
#include "module.h"
#include "dcompute/codegenvisitor.h"
DComputeTarget::DComputeTarget(llvm::LLVMContext &c, int v) : ctx(c) , tversion(v)
{    
}

void DComputeTarget::doCodeGen(Module* m) {
    
    // process module members
    for (unsigned k = 0; k < m->members->dim; k++) {
        Dsymbol *dsym = (*m->members)[k];
        assert(dsym);
        DcomputeDeclaration_codegen(dsym,_ir,*this);
    }
    
    if (global.errors) {
        fatal();
    }
}


void DComputeTarget::emit(Module* m) {
    doCodeGen(m);
    runReflectPass();
}