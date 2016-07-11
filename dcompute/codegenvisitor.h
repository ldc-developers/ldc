//
//  codegenvisitor.h
//  ldc
//
//  Created by Nicholas Wilson on 11/07/2016.
//
//

#ifndef LDC_DCOMPUTE_CODEGENVISTOR_H
#define LDC_DCOMPUTE_CODEGENVISTOR_H
#include "Visitor.h"
#include "gen/irstate.h"
#include "dcompute/target.h"
void DcomputeDeclaration_codegen(Dsymbol *decl, IRState *irs, DComputeTarget &dct);
#endif 
