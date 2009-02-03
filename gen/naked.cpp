#include "gen/llvm.h"

#include "expression.h"
#include "statement.h"
#include "declaration.h"

#include <cassert>

#include "gen/logger.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"

//////////////////////////////////////////////////////////////////////////////////////////

void Statement::toNakedIR(IRState *p)
{
    error("not allowed in naked function");
}

//////////////////////////////////////////////////////////////////////////////////////////

void CompoundStatement::toNakedIR(IRState *p)
{
    Logger::println("CompoundStatement::toNakedIR(): %s", loc.toChars());
    LOG_SCOPE;

    if (statements)
    for (unsigned i = 0; i < statements->dim; i++)
    {
        Statement* s = (Statement*)statements->data[i];
        if (s) s->toNakedIR(p);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void ExpStatement::toNakedIR(IRState *p)
{
    Logger::println("ExpStatement::toNakedIR(): %s", loc.toChars());
    LOG_SCOPE;

    // only expstmt supported in declarations
    if (exp->op != TOKdeclaration)
    {
        Statement::toNakedIR(p);
        return;
    }

    DeclarationExp* d = (DeclarationExp*)exp;
    VarDeclaration* vd = d->declaration->isVarDeclaration();
    FuncDeclaration* fd = d->declaration->isFuncDeclaration();
    EnumDeclaration* ed = d->declaration->isEnumDeclaration();

    // and only static variable/function declaration
    // no locals or nested stuffies!
    if (!vd && !fd && !ed)
    {
        Statement::toNakedIR(p);
        return;
    }
    else if (vd && !vd->isDataseg())
    {
        error("non-static variable '%s' not allowed in naked function", vd->toChars());
        return;
    }
    else if (fd && !fd->isStatic())
    {
        error("non-static nested function '%s' not allowed in naked function", fd->toChars());
        return;
    }
    // enum decls should always be safe

    // make sure the symbols gets processed
    d->declaration->toObjFile(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

void LabelStatement::toNakedIR(IRState *p)
{
    Logger::println("LabelStatement::toNakedIR(): %s", loc.toChars());
    LOG_SCOPE;

    p->nakedAsm << p->func()->decl->mangle() << "_" << ident->toChars() << ":";

    if (statement)
        statement->toNakedIR(p);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineNakedFunction(FuncDeclaration* fd)
{
    Logger::println("DtoDefineNakedFunction(%s)", fd->mangle());
    LOG_SCOPE;

    assert(fd->ir.irFunc);
    gIR->functions.push_back(fd->ir.irFunc);

    // we need to do special processing on the body, since we only want
    // to allow actual inline asm blocks to reach the final asm output

    std::ostringstream& asmstr = gIR->nakedAsm;

    // build function header

    // REALLY FIXME: this is most likely extremely platform dependent

    const char* mangle = fd->mangle();
    const char* linkage = "globl";
    std::string section = "text";
    unsigned align = 16;

    std::ostringstream tmpstr;

    if (DtoIsTemplateInstance(fd))
    {
        linkage = "weak";
        tmpstr << "section\t.gnu.linkonce.t." << mangle << ",\"ax\",@progbits";
        section = tmpstr.str();
    }

    asmstr << "\t." << section << std::endl;
    asmstr << "\t.align\t" << align << std::endl;
    asmstr << "\t." << linkage << "\t" << mangle << std::endl;
    asmstr << "\t.type\t" << mangle << ",@function" << std::endl;
    asmstr << mangle << ":" << std::endl;

    // emit body
    fd->fbody->toNakedIR(gIR);

    // emit size after body
    // why? dunno, llvm seems to do it by default ..
    asmstr << "\t.size\t" << mangle << ", .-" << mangle << std::endl << std::endl;

    gIR->module->appendModuleInlineAsm(asmstr.str());
    asmstr.str("");

    gIR->functions.pop_back();
}
