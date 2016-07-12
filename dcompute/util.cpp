//
//  util.cpp
//  ldc
//
//  Created by Nicholas Wilson on 12/07/2016.
//
//

#include "util.h"

#include "dsymbol.h"
#include "module.h"
#include "identifier.h"
#include "gen/logger.h"
bool isFromDCompute_Types(Dsymbol *sym)
{
    IF_LOG Logger::println("isFromDCompute_Types(%s)", sym->toPrettyChars());
    auto moduleDecl = sym->getModule()->md;

    if (moduleDecl->packages->dim != 2)
        return false;
    if (strcmp("dcompute", (*moduleDecl->packages)[0]->string))
        return false;
    if (strcmp("types", (*moduleDecl->packages)[1]->string))
        return false;
    return true;
}


