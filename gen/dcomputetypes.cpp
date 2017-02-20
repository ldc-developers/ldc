//===-- gen/dcomputetypes.cpp
//----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcomputetypes.h"
#include "gen/logger.h"
#include "ddmd/dsymbol.h"
#include "ddmd/module.h"
#include "ddmd/identifier.h"
#include "ddmd/template.h"
#include "ddmd/declaration.h"
#include "ddmd/aggregate.h"

bool isFromLDC_DComputeTypes(Dsymbol *sym) {
  IF_LOG Logger::println("isFromLDC_DComputeTypes(%s)", sym->toPrettyChars());
  LOG_SCOPE
  auto mod = sym->getModule();
  if (!mod)
    return false;
  IF_LOG Logger::println("mod = %s", mod->toPrettyChars());
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;
  if (!moduleDecl->packages)
    return false;
  if (moduleDecl->packages->dim != 2)
    return false;
  if (strcmp("ldc", (*moduleDecl->packages)[0]->string))
    return false;
  if (strcmp("dcomputetypes", (*moduleDecl->packages)[1]->string))
    return false;
  return true;
}

DcomputePointer::DcomputePointer(StructDeclaration *sd)
{
  if (!isFromLDC_DComputeTypes(sd) || strcmp(sd->ident->string, "Pointer")) {
    addrspace = -1;
    type = nullptr;
    return;
  }

  TemplateInstance *ti = sd->isInstantiated();
  addrspace = isExpression((*ti->tiargs)[0])->toInteger();
  type = isType((*ti->tiargs)[1]);
}

DcomputePointer::DcomputePointer()
{
  addrspace = -1;
  type = nullptr;
}
