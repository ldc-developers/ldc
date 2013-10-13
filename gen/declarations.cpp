//===-- declarations.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "aggregate.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "init.h"
#include "rmem.h"
#include "template.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/ir.h"
#include "ir/irtype.h"
#include "ir/irvar.h"

/* ================================================================== */

void Dsymbol::codegen(Ir*)
{
    IF_LOG Logger::println("Ignoring Dsymbol::codegen for %s", toPrettyChars());
}

/* ================================================================== */

void InterfaceDeclaration::codegen(Ir* p)
{
    IF_LOG Logger::println("InterfaceDeclaration::codegen: '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
    {
        DtoResolveDsymbol(this);

        // Emit any members (e.g. final functions).
        for (ArrayIter<Dsymbol> it(members); !it.done(); it.next())
        {
            it->codegen(p);
        }

        // Emit TypeInfo.
        DtoTypeInfoOf(type);

        // Define __InterfaceZ.
        llvm::GlobalVariable *interfaceZ = ir.irAggr->getClassInfoSymbol();
        interfaceZ->setInitializer(ir.irAggr->getClassInfoInit());
        interfaceZ->setLinkage(DtoExternalLinkage(this));
    }
}

/* ================================================================== */

void StructDeclaration::codegen(Ir* p)
{
    IF_LOG Logger::println("StructDeclaration::codegen: '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
    {
        DtoResolveStruct(this);

        for (ArrayIter<Dsymbol> it(members); !it.done(); it.next())
        {
            it->codegen(p);
        }

        // Define the __initZ symbol.
        llvm::GlobalVariable *initZ = ir.irAggr->getInitSymbol();
        initZ->setInitializer(ir.irAggr->getDefaultInit());
        initZ->setLinkage(DtoExternalLinkage(this));

        // emit typeinfo
        DtoTypeInfoOf(type);
    }
}

/* ================================================================== */

void ClassDeclaration::codegen(Ir* p)
{
    IF_LOG Logger::println("ClassDeclaration::codegen: '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
    {
        DtoResolveClass(this);

        for (ArrayIter<Dsymbol> it(members); !it.done(); it.next())
        {
            it->codegen(p);
        }

        llvm::GlobalValue::LinkageTypes const linkage = DtoExternalLinkage(this);

        llvm::GlobalVariable *initZ = ir.irAggr->getInitSymbol();
        initZ->setInitializer(ir.irAggr->getDefaultInit());
        initZ->setLinkage(linkage);

        llvm::GlobalVariable *vtbl = ir.irAggr->getVtblSymbol();
        vtbl->setInitializer(ir.irAggr->getVtblInit());
        vtbl->setLinkage(linkage);

        llvm::GlobalVariable *classZ = ir.irAggr->getClassInfoSymbol();
        classZ->setInitializer(ir.irAggr->getClassInfoInit());
        classZ->setLinkage(linkage);

        // No need to do TypeInfo here, it is <name>__classZ for classes in D2.
    }
}

/* ================================================================== */

void TupleDeclaration::codegen(Ir* p)
{
    IF_LOG Logger::println("TupleDeclaration::codegen(): '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    assert(isexp);
    assert(objects);

    int n = objects->dim;

    for (int i=0; i < n; ++i)
    {
        DsymbolExp* exp = static_cast<DsymbolExp*>(objects->data[i]);
        assert(exp->op == TOKdsymbol);
        exp->s->codegen(p);
    }
}

/* ================================================================== */

void VarDeclaration::codegen(Ir* p)
{
    IF_LOG Logger::println("VarDeclaration::codegen(): '%s'", toPrettyChars());
    LOG_SCOPE;

    if (ir.defined) return;
    ir.defined = true;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    DtoResolveVariable(this);

    // just forward aliases
    if (aliassym)
    {
        Logger::println("alias sym");
        toAlias()->codegen(p);
        return;
    }

    // global variable
    if (isDataseg() || (storage_class & (STCconst | STCimmutable) && init))
    {
        Logger::println("data segment");

    #if 0 // TODO:
        assert(!(storage_class & STCmanifest) &&
            "manifest constant being codegen'd!");
    #endif

        llvm::GlobalVariable *gvar = llvm::cast<llvm::GlobalVariable>(
            this->ir.irGlobal->value);
        assert(gvar && "DtoResolveVariable should have created value");

        const llvm::GlobalValue::LinkageTypes llLinkage = DtoLinkage(this);

        // Check if we are defining or just declaring the global in this module.
        if (!(storage_class & STCextern))
        {
            // Build the initializer. Might use this->ir.irGlobal->value!
            LLConstant *initVal = DtoConstInitializer(loc, type, init);

            // In case of type mismatch, swap out the variable.
            if (initVal->getType() != gvar->getType()->getElementType())
            {
                llvm::GlobalVariable* newGvar = getOrCreateGlobal(loc,
                    *gIR->module, initVal->getType(), gvar->isConstant(),
                    llLinkage, 0,
                    "", // We take on the name of the old global below.
                    gvar->isThreadLocal());

                newGvar->setAlignment(gvar->getAlignment());
                newGvar->takeName(gvar);

                llvm::Constant* newValue =
                    llvm::ConstantExpr::getBitCast(newGvar, gvar->getType());
                gvar->replaceAllUsesWith(newValue);

                gvar->eraseFromParent();
                gvar = newGvar;
                this->ir.irGlobal->value = newGvar;
            }

            // Now, set the initializer.
            assert(!ir.irGlobal->constInit);
            ir.irGlobal->constInit = initVal;
            gvar->setInitializer(initVal);
            gvar->setLinkage(llLinkage);

            // Also set up the edbug info.
            gIR->DBuilder.EmitGlobalVariable(gvar, this);
        }

        // If this global is used from a naked function, we need to create an
        // artificial "use" for it, or it could be removed by the optimizer if
        // the only reference to it is in inline asm.
        if (nakedUse)
            gIR->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));

        if (Logger::enabled())
            Logger::cout() << *gvar << '\n';
    }
}

/* ================================================================== */

void TypedefDeclaration::codegen(Ir*)
{
    IF_LOG Logger::println("TypedefDeclaration::codegen: '%s'", toPrettyChars());
    LOG_SCOPE;

    if (ir.defined) return;
    ir.defined = true;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    // generate typeinfo
    DtoTypeInfoOf(type, false);
}

/* ================================================================== */

void EnumDeclaration::codegen(Ir*)
{
    IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'", toPrettyChars());

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }
}

/* ================================================================== */

void FuncDeclaration::codegen(Ir* p)
{
    // don't touch function aliases, they don't contribute any new symbols
    if (!isFuncAliasDeclaration())
    {
        DtoDefineFunction(this);
    }
}

/* ================================================================== */

void TemplateInstance::codegen(Ir* p)
{
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    if (!errors && members)
    {
        for (unsigned i = 0; i < members->dim; i++)
        {
            (*members)[i]->codegen(p);
        }
    }
}

/* ================================================================== */

void TemplateMixin::codegen(Ir* p)
{
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'", toPrettyChars());
    LOG_SCOPE

    if (ir.defined) return;
    ir.defined = true;

    if (!errors && members)
    {
        for (unsigned i = 0; i < members->dim; i++)
        {
            (*members)[i]->codegen(p);
        }
    }
}

/* ================================================================== */

void AttribDeclaration::codegen(Ir* p)
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = static_cast<Dsymbol *>(d->data[i]);
            s->codegen(p);
        }
    }
}

/* ================================================================== */

void PragmaDeclaration::codegen(Ir* p)
{
    if (ident == Id::lib)
    {
        assert(args && args->dim == 1);

        Expression *e = static_cast<Expression *>(args->data[0]);

        assert(e->op == TOKstring);
        StringExp *se = static_cast<StringExp *>(e);

        size_t nameLen = se->len;
        if (global.params.targetTriple.getOS() == llvm::Triple::MinGW32)
        {
            if (nameLen > 4 &&
                !memcmp(static_cast<char*>(se->string) + nameLen - 4, ".lib", 4))
            {
                // On MinGW, strip the .lib suffix, if any, to improve
                // compatibility with code written for DMD (we pass the name to GCC
                // via -l, just as on Posix).
                nameLen -= 4;
            }

            if (nameLen >= 7 && !memcmp(se->string, "shell32", 7))
            {
                // Another DMD compatibility kludge: Ignore
                // pragma(lib, "shell32.lib"), it is implicitly provided by
                // MinGW.
                return;
            }
        }

        size_t const n = nameLen + 3;
        char *arg = static_cast<char *>(mem.malloc(n));
        arg[0] = '-';
        arg[1] = 'l';
        memcpy(arg + 2, se->string, nameLen);
        arg[n-1] = 0;
        global.params.linkswitches->push(arg);
    }
    AttribDeclaration::codegen(p);
}

/* ================================================================== */
