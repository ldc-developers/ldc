#include "gen/llvm.h"
#include "llvm/InlineAsm.h"

#include "expression.h"
#include "statement.h"
#include "declaration.h"
#include "template.h"

#include <cassert>

#include "gen/logger.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/dvalue.h"

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
    d->declaration->codegen(Type::sir);
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

    // FIXME: could we perhaps use llvm asmwriter to give us these details ?

    const char* mangle = fd->mangle();
    std::ostringstream tmpstr;

    // osx is different
    // also mangling has an extra underscore prefixed
    if (global.params.os == OSMacOSX)
    {
        std::string section = "text";
        bool weak = false;
        if (DtoIsTemplateInstance(fd))
        {
            tmpstr << "section\t__TEXT,__textcoal_nt,coalesced,pure_instructions";
            section = tmpstr.str();
            weak = true;
        }
        asmstr << "\t." << section << std::endl;
        asmstr << "\t.align\t4,0x90" << std::endl;
        asmstr << "\t.globl\t_" << mangle << std::endl;
        if (weak)
        {
            asmstr << "\t.weak_definition\t_" << mangle << std::endl;
        }
        asmstr << "_" << mangle << ":" << std::endl;
    }
    // this works on linux x86 32 and 64 bit
    // assume it works everywhere else as well for now
    else
    {
        const char* linkage = "globl";
        std::string section = "text";
        if (DtoIsTemplateInstance(fd))
        {
            linkage = "weak";
            tmpstr << "section\t.gnu.linkonce.t." << mangle << ",\"ax\",@progbits";
            section = tmpstr.str();
        }
        asmstr << "\t." << section << std::endl;
        asmstr << "\t.align\t16" << std::endl;
        asmstr << "\t." << linkage << "\t" << mangle << std::endl;
        asmstr << "\t.type\t" << mangle << ",@function" << std::endl;
        asmstr << mangle << ":" << std::endl;
    }

    // emit body
    fd->fbody->toNakedIR(gIR);

    // emit size after body
    // llvm does this on linux, but not on osx
    if (global.params.os != OSMacOSX)
    {
        asmstr << "\t.size\t" << mangle << ", .-" << mangle << std::endl << std::endl;
    }

    gIR->module->appendModuleInlineAsm(asmstr.str());
    asmstr.str("");

    gIR->functions.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

void emitABIReturnAsmStmt(IRAsmBlock* asmblock, Loc loc, FuncDeclaration* fdecl)
{
    Logger::println("emitABIReturnAsmStmt(%s)", fdecl->mangle());
    LOG_SCOPE;

    IRAsmStmt* as = new IRAsmStmt;

    const LLType* llretTy = DtoType(fdecl->type->nextOf());
    asmblock->retty = llretTy;
    asmblock->retn = 1;

    // FIXME: This should probably be handled by the TargetABI somehow.
    //        It should be able to do this for a greater variety of types.

    // x86
    if (global.params.cpu == ARCHx86)
    {
        LINK l = fdecl->linkage;
        assert((l == LINKd || l == LINKc || l == LINKwindows) && "invalid linkage for asm implicit return");

        Type* rt = fdecl->type->nextOf()->toBasetype();
        if (rt->isintegral() || rt->ty == Tpointer || rt->ty == Tclass || rt->ty == Taarray)
        {
            if (rt->size() == 8) {
                as->out_c = "=A,";
            } else {
                as->out_c = "={ax},";
            }
        }
        else if (rt->isfloating())
        {
            if (rt->iscomplex()) {
                if (fdecl->linkage == LINKd) {
                    // extern(D) always returns on the FPU stack
                    as->out_c = "={st},={st(1)},";
                    asmblock->retn = 2;
                } else if (rt->ty == Tcomplex32) {
                    // extern(C) cfloat is return as i64
                    as->out_c = "=A,";
                    asmblock->retty = LLType::Int64Ty;
                } else {
                    // cdouble and creal extern(C) are returned in pointer
                    // don't add anything!
                    asmblock->retty = LLType::VoidTy;
                    asmblock->retn = 0;
                    return;
                }
            } else {
                as->out_c = "={st},";
            }
        }
        else if (rt->ty == Tarray || rt->ty == Tdelegate)
        {
            as->out_c = "={ax},={dx},";
            asmblock->retn = 2;
        #if 0
            // this is to show how to allocate a temporary for the return value
            // in case the appropriate multi register constraint isn't supported.
            // this way abi return from inline asm can still be emulated.
            // note that "$<<out0>>" etc in the asm will translate to the correct
            // numbered output when the asm block in finalized

            // generate asm
            as->out_c = "=*m,=*m,";
            LLValue* tmp = DtoAlloca(llretTy, ".tmp_asm_ret");
            as->out.push_back( tmp );
            as->out.push_back( DtoGEPi(tmp, 0,1) );
            as->code = "movd %eax, $<<out0>>" "\n\t" "mov %edx, $<<out1>>";

            // fix asmblock
            asmblock->retn = 0;
            asmblock->retemu = true;
            asmblock->asmBlock->abiret = tmp;

            // add "ret" stmt at the end of the block
            asmblock->s.push_back(as);

            // done, we don't want anything pushed in the front of the block
            return;
        #endif
        }
        else
        {
            error(loc, "unimplemented return type '%s' for implicit abi return", rt->toChars());
            fatal();
        }
    }

    // x86_64
    else if (global.params.cpu == ARCHx86_64)
    {
        LINK l = fdecl->linkage;
        /* TODO: Check if this works with extern(Windows), completely untested.
         *       In particular, returning cdouble may not work with
         *       extern(Windows) since according to X86CallingConv.td it
         *       doesn't allow XMM1 to be used.
         * (So is extern(C), but that should be fine as the calling convention
         * is identical to that of extern(D))
         */
        assert((l == LINKd || l == LINKc || l == LINKwindows) && "invalid linkage for asm implicit return");

        Type* rt = fdecl->type->nextOf()->toBasetype();
        if (rt->isintegral() || rt->ty == Tpointer || rt->ty == Tclass || rt->ty == Taarray)
        {
            as->out_c = "={ax},";
        }
        else if (rt->isfloating())
        {
            if (rt == Type::tcomplex80) {
                // On x87 stack, re=st, im=st(1)
                as->out_c = "={st},={st(1)},";
                asmblock->retn = 2;
            } else if (rt == Type::tfloat80 || rt == Type::timaginary80) {
                // On x87 stack
                as->out_c = "={st},";
            } else if (l != LINKd && rt == Type::tcomplex32) {
                // LLVM and GCC disagree on how to return {float, float}.
                // For compatibility, use the GCC/LLVM-GCC way for extern(C/Windows)
                // extern(C) cfloat -> %xmm0 (extract two floats)
                as->out_c = "={xmm0},";
                asmblock->retty = LLType::DoubleTy;
            } else if (rt->iscomplex()) {
                // cdouble and extern(D) cfloat -> re=%xmm0, im=%xmm1
                as->out_c = "={xmm0},={xmm1},";
                asmblock->retn = 2;
            } else {
                // Plain float/double/ifloat/idouble
                as->out_c = "={xmm0},";
            }
        }
        else if (rt->ty == Tarray || rt->ty == Tdelegate)
        {
            as->out_c = "={ax},={dx},";
            asmblock->retn = 2;
        }
        else
        {
            error(loc, "unimplemented return type '%s' for implicit abi return", rt->toChars());
            fatal();
        }
    }

    // unsupported
    else
    {
        error(loc, "this target (%s) does not implement inline asm falling off the end of the function", global.params.targetTriple);
        fatal();
    }

    // return values always go in the front
    asmblock->s.push_front(as);
}

//////////////////////////////////////////////////////////////////////////////////////////

// sort of kinda related to naked ...

DValue * DtoInlineAsmExpr(Loc loc, FuncDeclaration * fd, Expressions * arguments)
{
    Logger::println("DtoInlineAsmExpr @ %s", loc.toChars());
    LOG_SCOPE;

    TemplateInstance* ti = fd->toParent()->isTemplateInstance();
    assert(ti && "invalid inline __asm expr");

    assert(arguments->dim >= 2 && "invalid __asm call");

    // get code param
    Expression* e = (Expression*)arguments->data[0];
    Logger::println("code exp: %s", e->toChars());
    StringExp* se = (StringExp*)e;
    if (e->op != TOKstring || se->sz != 1)
    {
        e->error("__asm code argument is not a char[] string literal");
        fatal();
    }
    std::string code((char*)se->string, se->len);

    // get constraints param
    e = (Expression*)arguments->data[1];
    Logger::println("constraint exp: %s", e->toChars());
    se = (StringExp*)e;
    if (e->op != TOKstring || se->sz != 1)
    {
        e->error("__asm constraints argument is not a char[] string literal");
        fatal();
    }
    std::string constraints((char*)se->string, se->len);

    // build runtime arguments
    size_t n = arguments->dim;

    LLSmallVector<llvm::Value*, 8> args;
    args.reserve(n-2);
    std::vector<const llvm::Type*> argtypes;
    argtypes.reserve(n-2);

    for (size_t i = 2; i < n; i++)
    {
        e = (Expression*)arguments->data[i];
        args.push_back(e->toElem(gIR)->getRVal());
        argtypes.push_back(args.back()->getType());
    }

    // build asm function type
    const llvm::Type* ret_type = DtoType(fd->type->nextOf());
    llvm::FunctionType* FT = llvm::FunctionType::get(ret_type, argtypes, false);

    // build asm call
    bool sideeffect = true;
    llvm::InlineAsm* ia = llvm::InlineAsm::get(FT, code, constraints, sideeffect);

    llvm::Value* v = gIR->ir->CreateCall(ia, args.begin(), args.end(), "");

    // return call as im value
    return new DImValue(fd->type->nextOf(), v);
}











