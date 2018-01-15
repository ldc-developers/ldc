//===-- gen/dcompute/druntime.cpp -----------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/druntime.h"
#include "ddmd/dsymbol.h"
#include "ddmd/module.h"
#include "ddmd/identifier.h"
#include "ddmd/template.h"
#include "ddmd/declaration.h"
#include "ddmd/aggregate.h"
#include "id.h"

#include "gen/logger.h"

bool isFromLDC_DCompute(Dsymbol *sym) {
  auto mod = sym->getModule();
  if (!mod)
    return false;
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;
  if (!moduleDecl->packages)
    return false;

  if (moduleDecl->packages->dim != 1)
    return false;
  if ((*moduleDecl->packages)[0] != Id::ldc)
    return false;

  return moduleDecl->id == Id::dcompute;
}

llvm::Optional<DcomputeAddrspacedType> toDcomputeAddrspacedType(VarDeclaration *vd) {
  StructDeclaration *sd = nullptr;
  if (vd->type->ty == Tstruct)
    sd = ((TypeStruct*)vd->type)->sym;
  return toDcomputeAddrspacedType(sd);
}

llvm::Optional<DcomputeAddrspacedType> toDcomputeAddrspacedType(StructDeclaration *sd) {
  if (!sd ||
      !(sd->ident == Id::dcPointer || sd->ident == Id::dcVariable) ||
      !isFromLDC_DCompute(sd))
  {
    return llvm::Optional<DcomputeAddrspacedType>(llvm::None);
  }
    
  TemplateInstance *ti = sd->isInstantiated();
  unsigned as = (unsigned)isExpression((*ti->tiargs)[0])->toInteger();
  Type *type = isType((*ti->tiargs)[1]);
  IF_LOG Logger::println("toDcomputeAddrspacedType(%s): %u : %s : %s",
                         sd->toPrettyChars(),as,type->toChars(),sd->ident->toChars());
  return llvm::Optional<DcomputeAddrspacedType>(DcomputeAddrspacedType(as, type,sd->ident));
}

unsigned addressSpaceForVarDeclaration(VarDeclaration *vd) {
    auto dcas = toDcomputeAddrspacedType(vd);
    unsigned as = 0;
    if (dcas && dcas->id == Id::dcVariable) {
      as = dcas->translate();
      IF_LOG Logger::println("addressSpaceForVarDeclaration: %s: as %u (was %u)",
                             vd->toChars(),as,dcas->addrspace);
    }

    return as;
}
