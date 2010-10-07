#include "gen/llvm.h"

#include "aggregate.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "mem.h"
#include "template.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"

#include "ir/ir.h"
#include "ir/irvar.h"
#include "ir/irtype.h"
#include "ir/irtypestruct.h"

/* ================================================================== */

void Dsymbol::codegen(Ir*)
{
    Logger::println("Ignoring Dsymbol::toObjFile for %s", toChars());
}

/* ================================================================== */

void Declaration::codegen(Ir*)
{
    Logger::println("Ignoring Declaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void InterfaceDeclaration::codegen(Ir*)
{
    //Logger::println("Ignoring InterfaceDeclaration::toObjFile for %s", toChars());
    DtoResolveDsymbol(this);
}

/* ================================================================== */

void StructDeclaration::codegen(Ir*)
{
    DtoResolveDsymbol(this);
}

/* ================================================================== */

void ClassDeclaration::codegen(Ir*)
{
    DtoResolveDsymbol(this);
}

/* ================================================================== */

void TupleDeclaration::codegen(Ir* p)
{
    Logger::println("TupleDeclaration::toObjFile(): %s", toChars());

    assert(isexp);
    assert(objects);

    int n = objects->dim;

    for (int i=0; i < n; ++i)
    {
        DsymbolExp* exp = (DsymbolExp*)objects->data[i];
        assert(exp->op == TOKdsymbol);
        exp->s->codegen(p);
    }
}

/* ================================================================== */

// FIXME: this is horrible!!!

void VarDeclaration::codegen(Ir* p)
{
    Logger::print("VarDeclaration::toObjFile(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

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

    #if DMDV2
        if (storage_class & STCmanifest)
        {
            assert(0 && "manifest constant being codegened!!!");
        }
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

        const LLType* _type = this->ir.irGlobal->type.get();
        llvm::GlobalValue::LinkageTypes _linkage = DtoLinkage(this);
        std::string _name(mangle());

        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module,_type,_isconst,_linkage,NULL,_name,0,isThreadlocal());
        this->ir.irGlobal->value = gvar;

        // set the alignment
        gvar->setAlignment(this->type->alignsize());

        if (Logger::enabled())
            Logger::cout() << *gvar << '\n';

        // if this global is used from a nested function, this is necessary or
        // optimization could potentially remove the global (if it's the only use)
        if (nakedUse)
            gIR->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));

        // initialize
        DtoConstInitGlobal(this);
    }
}

/* ================================================================== */

void TypedefDeclaration::codegen(Ir*)
{
    Logger::print("TypedefDeclaration::toObjFile: %s\n", toChars());
    LOG_SCOPE;

    // generate typeinfo
    DtoTypeInfoOf(type, false);
}

/* ================================================================== */

void EnumDeclaration::codegen(Ir*)
{
    Logger::println("Ignoring EnumDeclaration::toObjFile for %s", toChars());
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
    printf("TemplateInstance::toObjFile('%s', this = %p)\n", toChars(), this);
#endif
    if (!errors && members)
    {
        for (int i = 0; i < members->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)members->data[i];
            s->codegen(p);
        }
    }
}

/* ================================================================== */

void TemplateMixin::codegen(Ir* p)
{
    TemplateInstance::codegen(p);
}

/* ================================================================== */

void AttribDeclaration::codegen(Ir* p)
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->codegen(p);
        }
    }
}

/* ================================================================== */

void obj_includelib(const char* lib);

void PragmaDeclaration::codegen(Ir* p)
{
    if (ident == Id::lib)
    {
        assert(args && args->dim == 1);

        Expression *e = (Expression *)args->data[0];

        assert(e->op == TOKstring);

        StringExp *se = (StringExp *)e;
        char *name = (char *)mem.malloc(se->len + 1);
        memcpy(name, se->string, se->len);
        name[se->len] = 0;
        obj_includelib(name);
    }
    AttribDeclaration::codegen(p);
}

/* ================================================================== */
