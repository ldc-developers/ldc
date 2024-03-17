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

using namespace dmd;

bool isFromLDC_Mod(Dsymbol *sym, Identifier* id) {
  auto mod = sym->getModule();
  if (!mod)
    return false;
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;

  if (moduleDecl->packages.length != 1)
    return false;
  if (moduleDecl->packages.ptr[0] != Id::ldc)
    return false;

  return moduleDecl->id == id;
}

bool isFromLDC_DCompute(Dsymbol *sym) {
  return isFromLDC_Mod(sym,Id::dcompute);
}
bool isFromLDC_OpenCL(Dsymbol *sym) {
  return isFromLDC_Mod(sym,Id::opencl);
}

llvm::Optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd) {
  if (sd->ident != Id::dcPointer || !isFromLDC_DCompute(sd)) {
#if LDC_LLVM_VER < 1600
    return llvm::Optional<DcomputePointer>(llvm::None);
#else
    return std::optional<DcomputePointer>(std::nullopt);
#endif
  }

  TemplateInstance *ti = sd->isInstantiated();
  int addrspace = isExpression((*ti->tiargs)[0])->toInteger();
  Type *type = isType((*ti->tiargs)[1]);
  return llvm::Optional<DcomputePointer>(DcomputePointer(addrspace, type));
}
