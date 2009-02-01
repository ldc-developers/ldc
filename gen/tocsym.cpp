
// Copyright (c) 1999-2005 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

// stubbed out for dmdfe. Original is in dmd/tocsym.c

#include <stddef.h>

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "declaration.h"
#include "statement.h"
#include "enum.h"
#include "aggregate.h"
#include "init.h"
#include "attrib.h"
#include "lexer.h"


Symbol *StaticStructInitDeclaration::toSymbol()
{
    return 0;
}

/*************************************
 * Helper
 */

Symbol *Dsymbol::toSymbolX(const char *prefix, int sclass, TYPE *t, const char *suffix)
{
  return 0;
}

/*************************************
 */

Symbol *Dsymbol::toSymbol()
{
  return 0;
}

/*********************************
 * Generate import symbol from symbol.
 */

Symbol *Dsymbol::toImport()
{
  return 0;
}

/*************************************
 */

Symbol *Dsymbol::toImport(Symbol *sym)
{
  return 0;
}

/*************************************
 */

Symbol *VarDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 */

Symbol *ClassInfoDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 */

Symbol *ModuleInfoDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 */

Symbol *TypeInfoDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 */

Symbol *FuncDeclaration::toSymbol()
{

    return 0;
}

/*************************************
 */

Symbol *FuncDeclaration::toThunkSymbol(int offset)
{
  return 0;
}

/*************************************
 */

Symbol *FuncAliasDeclaration::toSymbol()
{

    return 0;
}


/****************************************
 * Create a static symbol we can hang DT initializers onto.
 */

Symbol *static_sym()
{
  return 0;
}

/*************************************
 * Create the "ClassInfo" symbol
 */

Symbol *ClassDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 * Create the "InterfaceInfo" symbol
 */

Symbol *InterfaceDeclaration::toSymbol()
{
  return 0;
}

/*************************************
 * Create the "ModuleInfo" symbol
 */

Symbol *Module::toSymbol()
{
  return 0;
}

/*************************************
 * This is accessible via the ClassData, but since it is frequently
 * needed directly (like for rtti comparisons), make it directly accessible.
 */

Symbol *ClassDeclaration::toVtblSymbol()
{
  return 0;
}

/**********************************
 * Create the static initializer for the struct/class.
 */

Symbol *AggregateDeclaration::toInitializer()
{
  return 0;
}


/******************************************
 */

Symbol *Module::toModuleAssert()
{
  return 0;
}

/******************************************
 */

Symbol *Module::toModuleArray()
{
  return 0;
}

/********************************************
 * Determine the right symbol to look up
 * an associative array element.
 * Input:
 *	flags	0	don't add value signature
 *		1	add value signature
 */

Symbol *TypeAArray::aaGetSymbol(const char *func, int flags)
{
  return 0;
}

