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
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/todebug.h"
#include "gen/tollvm.h"
#include "ir/ir.h"
#include "ir/irtype.h"
#include "ir/irtypestruct.h"
#include "ir/irvar.h"

/* ================================================================== */

void Dsymbol::codegen(Ir*)
{
    Logger::println("Ignoring Dsymbol::codegen for %s", toChars());
}

/* ================================================================== */

void Declaration::codegen(Ir*)
{
    Logger::println("Ignoring Declaration::codegen for %s", toChars());
}

/* ================================================================== */

void InterfaceDeclaration::codegen(Ir*)
{
    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
        DtoResolveDsymbol(this);
}

/* ================================================================== */

void StructDeclaration::codegen(Ir*)
{
    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
        DtoResolveDsymbol(this);
}

/* ================================================================== */

void ClassDeclaration::codegen(Ir*)
{
    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    if (members && symtab)
        DtoResolveDsymbol(this);
}

/* ================================================================== */

void TupleDeclaration::codegen(Ir* p)
{
    Logger::println("TupleDeclaration::codegen(): %s", toChars());

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

// FIXME: this is horrible!!!

void VarDeclaration::codegen(Ir* p)
{
    Logger::print("VarDeclaration::codegen(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (type->ty == Terror)
    {   error("had semantic errors when compiling");
        return;
    }

    // just forward aliases
    if (aliassym)
    {
        Logger::println("alias sym");
        toAlias()->codegen(p);
        return;
    }

    // output the parent aggregate first
    if (AggregateDeclaration* ad = isMember())
        ad->codegen(p);

    // global variable
#if DMDV2
    // taken from dmd2/structs
    if (isDataseg() || (storage_class & (STCconst | STCimmutable) && init))
#else
    if (isDataseg())
#endif
    {
        Logger::println("data segment");

    #if DMDV2 && 0 // TODO:
        assert(!(storage_class & STCmanifest) &&
            "manifest constant being codegen'd!");
    #endif

        // don't duplicate work
        if (this->ir.resolved) return;
        this->ir.resolved = true;
        this->ir.declared = true;

        this->ir.irGlobal = new IrGlobal(this);

        Logger::println("parent: %s (%s)", parent->toChars(), parent->kind());

    #if DMDV2
        // not sure why this is only needed for d2
        bool _isconst = isConst() && init;
    #else
        bool _isconst = isConst();
    #endif

        Logger::println("Creating global variable");

        assert(!ir.initialized);
        ir.initialized = gIR->dmodule;
        std::string _name(mangle());

        LLType *_type = DtoConstInitializerType(type, init);

        // create the global variable
#if LDC_LLVM_VER >= 302
        // FIXME: clang uses a command line option for the thread model
        LLGlobalVariable* gvar = new LLGlobalVariable(*gIR->module, _type, _isconst,
                                                      DtoLinkage(this), NULL, _name, 0,
                                                      isThreadlocal() ? LLGlobalVariable::GeneralDynamicTLSModel
                                                                      : LLGlobalVariable::NotThreadLocal);
#else
        LLGlobalVariable* gvar = new LLGlobalVariable(*gIR->module, _type, _isconst,
                                                      DtoLinkage(this), NULL, _name, 0, isThreadlocal());
#endif
        this->ir.irGlobal->value = gvar;

        // Set the alignment (it is important not to use type->alignsize because
        // VarDeclarations can have an align() attribute independent of the type
        // as well).
        if (alignment != STRUCTALIGN_DEFAULT)
            gvar->setAlignment(alignment);

        if (Logger::enabled())
            Logger::cout() << *gvar << '\n';

        // if this global is used from a nested function, this is necessary or
        // optimization could potentially remove the global (if it's the only use)
        if (nakedUse)
            gIR->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));

        // assign the initializer
        if (!(storage_class & STCextern) && mustDefineSymbol(this))
        {
            if (Logger::enabled())
            {
                Logger::println("setting initializer");
                Logger::cout() << "global: " << *gvar << '\n';
    #if 0
                Logger::cout() << "init:   " << *initVal << '\n';
    #endif
            }
            // build the initializer
            LLConstant *initVal = DtoConstInitializer(loc, type, init);

            // set the initializer
            assert(!ir.irGlobal->constInit);
            ir.irGlobal->constInit = initVal;
            gvar->setInitializer(initVal);

            // do debug info
            DtoDwarfGlobalVariable(gvar, this);
        }
    }
}

/* ================================================================== */

void TypedefDeclaration::codegen(Ir*)
{
    Logger::print("TypedefDeclaration::codegen: %s\n", toChars());
    LOG_SCOPE;

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
    Logger::println("Ignoring EnumDeclaration::codegen for %s", toChars());

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
        DtoResolveDsymbol(this);
    }
}

/* ================================================================== */

void TemplateInstance::codegen(Ir* p)
{
#if LOG
    printf("TemplateInstance::codegen('%s', this = %p)\n", toChars(), this);
#endif
#if DMDV2
    if (ignore)
        return;
#endif

    if (!errors && members)
    {
        for (unsigned i = 0; i < members->dim; i++)
        {
            Dsymbol *s = static_cast<Dsymbol *>(members->data[i]);
            s->codegen(p);
        }
    }
}

/* ================================================================== */

void TemplateMixin::codegen(Ir* p)
{
    if (!errors && members)
    {
        for (unsigned i = 0; i < members->dim; i++)
        {
            Dsymbol *s = static_cast<Dsymbol *>(members->data[i]);
            if (s->isVarDeclaration())
                continue;
            s->codegen(p);
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
        char *name = static_cast<char *>(mem.malloc(se->len + 1));
        memcpy(name, se->string, se->len);
        name[se->len] = 0;

        size_t n = strlen(name)+3;
        char *arg = static_cast<char *>(mem.malloc(n));
        strcpy(arg, "-l");
        strncat(arg, name, n);
        global.params.linkswitches->push(arg);
    }
    AttribDeclaration::codegen(p);
}

/* ================================================================== */
