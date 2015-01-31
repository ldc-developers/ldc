//===-- irvar.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irmetadata.h"
#include "ir/irvar.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

llvm::Value *getIrValue(VarDeclaration *decl)
{
    return getIrVar(decl)->value;
}