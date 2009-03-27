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

void VarDeclaration::codegen(Ir* p)
{
    Logger::print("VarDeclaration::toObjFile(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (aliassym)
    {
        Logger::println("alias sym");
        toAlias()->codegen(p);
        return;
    }

    // global variable or magic
#if DMDV2
    // taken from dmd2/structs
    if (isDataseg() || (storage_class & (STCconst | STCinvariant) && init))
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

        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_type,_isconst,_linkage,NULL,_name,gIR->module);
        this->ir.irGlobal->value = gvar;

        if (Logger::enabled())
            Logger::cout() << *gvar << '\n';

        // if this global is used from a nested function, this is necessary or
        // optimization could potentially remove the global (if it's the only use)
        if (nakedUse)
            gIR->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));

        gIR->constInitList.push_back(this);
    }
    else
    {
        // might already have its irField, as classes derive each other without getting copies of the VarDeclaration
        if (!ir.irField)
        {
            assert(!ir.isSet());
            ir.irField = new IrField(this);
        }
        IrStruct* irstruct = gIR->topstruct();
        irstruct->addVar(this);

        Logger::println("added offset %u", offset);
    }
}

/* ================================================================== */

void TypedefDeclaration::codegen(Ir*)
{
    static int tdi = 0;
    Logger::print("TypedefDeclaration::toObjFile(%d): %s\n", tdi++, toChars());
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

void FuncDeclaration::codegen(Ir*)
{
    DtoResolveDsymbol(this);
}

/* ================================================================== */

void AnonDeclaration::codegen(Ir* p)
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        // get real aggregate parent
        IrStruct* irstruct = gIR->topstruct();

        // push a block on the stack
        irstruct->pushAnon(isunion);

        // go over children
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->codegen(p);
        }

        // finish
        irstruct->popAnon();
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
