//===-- gen/dcompute/druntime.cpp -----------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/druntime.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/module.h"
#include "dmd/template.h"

bool isFromLDC_DCompute(Dsymbol *sym) {
  auto mod = sym->getModule();
  if (!mod)
    return false;
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;
  if (!moduleDecl->packages)
    return false;

  if (moduleDecl->packages->length != 1)
    return false;
  if ((*moduleDecl->packages)[0] != Id::ldc)
    return false;

  return moduleDecl->id == Id::dcompute;
}

llvm::Optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd) {
  if (sd->ident != Id::dcPointer || !isFromLDC_DCompute(sd))
    return llvm::Optional<DcomputePointer>(llvm::None);

  TemplateInstance *ti = sd->isInstantiated();
  int addrspace = isExpression((*ti->tiargs)[0])->toInteger();
  Type *type = isType((*ti->tiargs)[1]);
  return llvm::Optional<DcomputePointer>(DcomputePointer(addrspace, type));
}
