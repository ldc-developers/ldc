//===-- dcompute/util.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "util.h"

#include "dsymbol.h"
#include "module.h"
#include "identifier.h"
#include "gen/logger.h"
bool isFromDCompute_Types(Dsymbol *sym) {
  IF_LOG Logger::println("isFromDCompute_Types(%s)", sym->toPrettyChars());
  LOG_SCOPE
  auto mod = sym->getModule();
  if (!mod)
    return false;
  IF_LOG Logger::println("mod = %s", mod->toPrettyChars());
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;
  IF_LOG Logger::println("moduleDecl = %p", moduleDecl);
  if (!moduleDecl->packages)
    return false;
  if (moduleDecl->packages->dim != 2)
    return false;

  IF_LOG Logger::println("moduleDecl.packages.dim = %d",
                         moduleDecl->packages->dim);
  if (strcmp("dcompute", (*moduleDecl->packages)[0]->string))
    return false;
  if (strcmp("types", (*moduleDecl->packages)[1]->string))
    return false;
  return true;
}

bool isFromDCompute_Attributes(Dsymbol *sym) {
  IF_LOG Logger::println("isFromDCompute_Attributes(%s)", sym->toPrettyChars());
  auto mod = sym->getModule();
  if (!mod)
    return false;
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;
  if (moduleDecl->packages->dim != 2)
    return false;
  if (strcmp("dcompute", (*moduleDecl->packages)[0]->string))
    return false;
  if (strcmp("attributes", (*moduleDecl->packages)[1]->string))
    return false;
  return true;
}
