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
#include "ddmd/dsymbol.h"
#include "ddmd/module.h"
#include "ddmd/identifier.h"
#include "gen/logger.h"
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
