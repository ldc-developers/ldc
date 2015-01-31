//===-- naked.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "expression.h"
#include "declaration.h"
#include "statement.h"
#include "template.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irmetadata.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/InlineAsm.h"
#else
#include "llvm/InlineAsm.h"
#endif
#include <cassert>

//////////////////////////////////////////////////////////////////////////////////////////
// FIXME: Integrate these functions
void AsmStatement_toNakedIR(AsmStatement *stmt, IRState *irs);

//////////////////////////////////////////////////////////////////////////////////////////

class ToNakedIRVisitor : public Visitor {
    IRState *irs;
public:

    ToNakedIRVisitor(IRState *irs) : irs(irs) { }

    //////////////////////////////////////////////////////////////////////////

    // Import all functions from class Visitor
    using Visitor::visit;

    //////////////////////////////////////////////////////////////////////////

    void visit(Statement *stmt) LLVM_OVERRIDE {
        error(Loc(), "Statement not allowed in naked function");
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AsmStatement *stmt) LLVM_OVERRIDE {
        AsmStatement_toNakedIR(stmt, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AsmBlockStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("AsmBlockStatement::toNakedIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        for (Statements::iterator I = stmt->statements->begin(),
                                  E = stmt->statements->end();
                                  I != E; ++I)
        {
            Statement *s = *I;
            if (s) s->accept(this);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(CompoundStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("CompoundStatement::toNakedIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        if (stmt->statements)
            for (Statements::iterator I = stmt->statements->begin(),
                                      E = stmt->statements->end();
                                      I != E; ++I)
            {
                Statement *s = *I;
                if (s) s->accept(this);
            }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ExpStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ExpStatement::toNakedIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // This happens only if there is a ; at the end:
        // asm { naked; ... };
        // Is this a legal AST?
        if (!stmt->exp) return;

        // only expstmt supported in declarations
        if (!stmt->exp || stmt->exp->op != TOKdeclaration)
        {
            visit(static_cast<Statement *>(stmt));
            return;
        }

        DeclarationExp* d = static_cast<DeclarationExp*>(stmt->exp);
        VarDeclaration* vd = d->declaration->isVarDeclaration();
        FuncDeclaration* fd = d->declaration->isFuncDeclaration();
        EnumDeclaration* ed = d->declaration->isEnumDeclaration();

        // and only static variable/function declaration
        // no locals or nested stuffies!
        if (!vd && !fd && !ed)
        {
            visit(static_cast<Statement *>(stmt));
            return;
        }
        else if (vd && !(vd->storage_class & (STCstatic | STCmanifest)))
        {
            error(vd->loc, "non-static variable '%s' not allowed in naked function", vd->toChars());
            return;
        }
        else if (fd && !fd->isStatic())
        {
            error(fd->loc, "non-static nested function '%s' not allowed in naked function", fd->toChars());
            return;
        }
        // enum decls should always be safe

        // make sure the symbols gets processed
        // TODO: codegen() here is likely incorrect
        Declaration_codegen(d->declaration, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(LabelStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("LabelStatement::toNakedIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        printLabelName(irs->nakedAsm, mangleExact(irs->func()->decl), stmt->ident->toChars());
        irs->nakedAsm << ":";

        if (stmt->statement)
            stmt->statement->accept(this);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineNakedFunction(FuncDeclaration* fd)
{
    IF_LOG Logger::println("DtoDefineNakedFunction(%s)", mangleExact(fd));
    LOG_SCOPE;

    gIR->functions.push_back(getIrFunc(fd));

    // we need to do special processing on the body, since we only want
    // to allow actual inline asm blocks to reach the final asm output

    std::ostringstream& asmstr = gIR->nakedAsm;

    // build function header

    // FIXME: could we perhaps use llvm asmwriter to give us these details ?

    const char* mangle = mangleExact(fd);
    std::ostringstream tmpstr;

    bool const isWin = global.params.targetTriple.isOSWindows();
    bool const isOSX = (global.params.targetTriple.getOS() == llvm::Triple::Darwin ||
        global.params.targetTriple.getOS() == llvm::Triple::MacOSX);

    // osx is different
    // also mangling has an extra underscore prefixed
    if (isOSX)
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
        asmstr << "\t.align\t4, 0x90" << std::endl;
        asmstr << "\t.globl\t_" << mangle << std::endl;
        if (weak)
        {
            asmstr << "\t.weak_definition\t_" << mangle << std::endl;
        }
        asmstr << "_" << mangle << ":" << std::endl;
    }
    // Windows is different
    else if (isWin)
    {
        std::string fullMangle;
#if LDC_LLVM_VER >= 305
        if (global.params.targetTriple.isWindowsGNUEnvironment())
#else
        if (global.params.targetTriple.getOS() == llvm::Triple::MinGW32)
#endif
        {
            fullMangle = "_";
        }
        fullMangle += mangle;

        asmstr << "\t.def\t" << fullMangle << ";" << std::endl;
        // hard code these two numbers for now since gas ignores .scl and llvm
        // is defaulting to .type 32 for everything I have seen
        asmstr << "\t.scl 2;" << std::endl;
        asmstr << "\t.type 32;" << std::endl;
        asmstr << "\t.endef" << std::endl;

        if (DtoIsTemplateInstance(fd))
        {
            asmstr << "\t.section\t.text$" << fullMangle << ",\"xr\"" << std::endl;
            asmstr << "\t.linkonce\tdiscard" << std::endl;
        }
        else
            asmstr << "\t.text" << std::endl;
        asmstr << "\t.globl\t" << fullMangle << std::endl;
        asmstr << "\t.align\t16, 0x90" << std::endl;
        asmstr << fullMangle << ":" << std::endl;
    }
    else
    {
        if (DtoIsTemplateInstance(fd))
        {
            asmstr << "\t.section\t.text." << mangle << ",\"axG\",@progbits,"
                   << mangle << ",comdat" << std::endl;
            asmstr << "\t.weak\t" << mangle << std::endl;
        }
        else
        {
            asmstr << "\t.text" << std::endl;
            asmstr << "\t.globl\t" << mangle << std::endl;
        }
        asmstr << "\t.align\t16, 0x90" << std::endl;
        asmstr << "\t.type\t" << mangle << ",@function" << std::endl;
        asmstr << mangle << ":" << std::endl;
    }

    // emit body
    ToNakedIRVisitor v(gIR);
    fd->fbody->accept(&v);

    // We could have generated new errors in toNakedIR(), but we are in codegen
    // already so we have to abort here.
    if (global.errors)
        fatal();

    // emit size after body
    // llvm does this on linux, but not on osx or Win
    if (!(isWin || isOSX))
    {
        asmstr << "\t.size\t" << mangle << ", .-" << mangle << std::endl << std::endl;
    }

    gIR->module->appendModuleInlineAsm(asmstr.str());
    asmstr.str("");

    gIR->functions.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

void emitABIReturnAsmStmt(IRAsmBlock* asmblock, Loc& loc, FuncDeclaration* fdecl)
{
    IF_LOG Logger::println("emitABIReturnAsmStmt(%s)", mangleExact(fdecl));
    LOG_SCOPE;

    IRAsmStmt* as = new IRAsmStmt;

    LLType* llretTy = DtoType(fdecl->type->nextOf());
    asmblock->retty = llretTy;
    asmblock->retn = 1;

    // FIXME: This should probably be handled by the TargetABI somehow.
    //        It should be able to do this for a greater variety of types.

    // x86
    if (global.params.targetTriple.getArch() == llvm::Triple::x86)
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
                    asmblock->retty = LLType::getInt64Ty(gIR->context());
                } else {
                    // cdouble and creal extern(C) are returned in pointer
                    // don't add anything!
                    asmblock->retty = LLType::getVoidTy(gIR->context());
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
            LLValue* tmp = DtoRawAlloca(llretTy, 0, ".tmp_asm_ret");
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
    else if (global.params.targetTriple.getArch() == llvm::Triple::x86_64)
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
                asmblock->retty = LLType::getDoubleTy(gIR->context());
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
        error(loc, "this target (%s) does not implement inline asm falling off the end of the function", global.params.targetTriple.str().c_str());
        fatal();
    }

    // return values always go in the front
    asmblock->s.push_front(as);
}

//////////////////////////////////////////////////////////////////////////////////////////

// sort of kinda related to naked ...

DValue * DtoInlineAsmExpr(Loc& loc, FuncDeclaration * fd, Expressions * arguments)
{
    IF_LOG Logger::println("DtoInlineAsmExpr @ %s", loc.toChars());
    LOG_SCOPE;

    TemplateInstance* ti = fd->toParent()->isTemplateInstance();
    assert(ti && "invalid inline __asm expr");

    assert(arguments->dim >= 2 && "invalid __asm call");

    // get code param
    Expression* e = (*arguments)[0];
    IF_LOG Logger::println("code exp: %s", e->toChars());
    StringExp* se = static_cast<StringExp*>(e);
    if (e->op != TOKstring || se->sz != 1)
    {
        e->error("__asm code argument is not a char[] string literal");
        fatal();
    }
    std::string code(static_cast<char*>(se->string), se->len);

    // get constraints param
    e = (*arguments)[1];
    IF_LOG Logger::println("constraint exp: %s", e->toChars());
    se = static_cast<StringExp*>(e);
    if (e->op != TOKstring || se->sz != 1)
    {
        e->error("__asm constraints argument is not a char[] string literal");
        fatal();
    }
    std::string constraints(static_cast<char*>(se->string), se->len);

    // build runtime arguments
    size_t n = arguments->dim;

    LLSmallVector<llvm::Value*, 8> args;
    args.reserve(n-2);
    std::vector<LLType*> argtypes;
    argtypes.reserve(n-2);

    for (size_t i = 2; i < n; i++)
    {
        args.push_back(toElem((*arguments)[i])->getRVal());
        argtypes.push_back(args.back()->getType());
    }

    // build asm function type
    Type* type = fd->type->nextOf()->toBasetype();
    LLType* ret_type = DtoType(type);
    llvm::FunctionType* FT = llvm::FunctionType::get(ret_type, argtypes, false);

    // build asm call
    bool sideeffect = true;
    llvm::InlineAsm* ia = llvm::InlineAsm::get(FT, code, constraints, sideeffect);

    llvm::Value* rv = gIR->ir->CreateCall(ia, args, "");

    // work around missing tuple support for users of the return value
    if (type->ty == Tstruct)
    {
        // make a copy
        llvm::Value* mem = DtoAlloca(type, ".__asm_tuple_ret");

        TypeStruct* ts = static_cast<TypeStruct*>(type);
        size_t n = ts->sym->fields.dim;
        for (size_t i = 0; i < n; i++)
        {
            llvm::Value* v = gIR->ir->CreateExtractValue(rv, i, "");
            llvm::Value* gep = DtoGEPi(mem, 0, i);
            DtoStore(v, gep);
        }

        return new DVarValue(fd->type->nextOf(), mem);
    }

    // return call as im value
    return new DImValue(fd->type->nextOf(), rv);
}











