//===-- asmstmt.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file originates from work by David Friedman for GDC released under
// the GPL 2 and Artistic licenses. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/InlineAsm.h"
#else
#include "llvm/InlineAsm.h"
#endif

//#include "d-gcc-includes.h"
//#include "total.h"
#include "mars.h"
#include "statement.h"
#include "scope.h"
#include "declaration.h"
#include "dsymbol.h"

#include <cassert>
#include <deque>
#include <cstring>
#include <string>
#include <sstream>

//#include "d-lang.h"
//#include "d-codegen.h"

#include "gen/irstate.h"
#include "gen/dvalue.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"
#include "gen/functions.h"

typedef enum {
    Arg_Integer,
    Arg_Pointer,
    Arg_Memory,
    Arg_FrameRelative,
    Arg_LocalSize,
    Arg_Dollar
} AsmArgType;

typedef enum {
    Mode_Input,
    Mode_Output,
    Mode_Update
} AsmArgMode;

struct AsmArg {
    Expression * expr;
    AsmArgType   type;
    AsmArgMode   mode;
    AsmArg(AsmArgType type, Expression * expr, AsmArgMode mode) {
        this->type = type;
        this->expr = expr;
        this->mode = mode;
    }
};

struct AsmCode {
    std::string insnTemplate;
    std::vector<AsmArg> args;
    std::vector<bool> regs;
    unsigned dollarLabel;
    int      clobbersMemory;
    AsmCode(int n_regs) {
        regs.resize(n_regs, false);
        dollarLabel = 0;
        clobbersMemory = 0;
    }
};

AsmStatement::AsmStatement(Loc loc, Token *tokens) :
    Statement(loc)
{
    this->tokens = tokens; // Do I need to copy these?
    asmcode = 0;
    asmalign = 0;
    refparam = 0;
    naked = 0;

    isBranchToLabel = NULL;
}

Statement *AsmStatement::syntaxCopy()
{
    // copy tokens? copy 'code'?
    AsmStatement * a_s = new AsmStatement(loc,tokens);
    a_s->asmcode = asmcode;
    a_s->refparam = refparam;
    a_s->naked = naked;
    return a_s;
}

struct AsmParserCommon
{
    virtual ~AsmParserCommon() {}
    virtual void run(Scope* sc, AsmStatement* asmst) = 0;
    virtual std::string getRegName(int i) = 0;
};
AsmParserCommon* asmparser = NULL;

#include "asm-x86.h" // x86 assembly parser
#define ASM_X86_64
#include "asm-x86.h" // x86_64 assembly parser
#undef ASM_X86_64

/**
 * Replaces <<func>> with the name of the currently codegen'd function.
 *
 * This kludge is required to handle labels correctly, as the instruction
 * strings for jumps, … are generated during semantic3, but attribute inference
 * might change the function type (and hence the mangled name) right at the end
 * of semantic3.
 */
static void replace_func_name(IRState* p, std::string& insnt)
{
    static const std::string needle("<<func>>");

    size_t pos;
    while (std::string::npos != (pos = insnt.find(needle)))
    {
        // This will only happen for few instructions, and only once for those.
        insnt.replace(pos, needle.size(), mangle(p->func()->decl));
    }
}

Statement* asmSemantic(AsmStatement *s, Scope *sc)
{
    if (sc->func && sc->func->isSafe())
        s->error("inline assembler not allowed in @safe function %s", sc->func->toChars());

    bool err = false;
    llvm::Triple const t = global.params.targetTriple;
    if (!(t.getArch() == llvm::Triple::x86 || t.getArch() == llvm::Triple::x86_64))
    {
        s->error("inline asm is not supported for the \"%s\" architecture",
            t.getArchName().str().c_str());
        err = true;
    }
    if (!global.params.useInlineAsm)
    {
        s->error("inline asm is not allowed when the -noasm switch is used");
        err = true;
    }
    if (err)
        fatal();

    //puts(toChars());

    sc->func->hasReturnExp |= 8;

    // empty statement -- still do the above things because they might be expected?
    if (!s->tokens)
        return s;

    if (!asmparser)
    {
        if (t.getArch() == llvm::Triple::x86)
            asmparser = new AsmParserx8632::AsmParser;
        else if (t.getArch() == llvm::Triple::x86_64)
            asmparser = new AsmParserx8664::AsmParser;
    }

    asmparser->run(sc, s);

    return s;
}

void AsmStatement_toIR(AsmStatement *stmt, IRState * irs)
{
    IF_LOG Logger::println("AsmStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    // sanity check
    assert(irs->func()->decl->hasReturnExp & 8);

    // get asm block
    IRAsmBlock* asmblock = irs->asmBlock;
    assert(asmblock);

    // debug info
    gIR->DBuilder.EmitStopPoint(stmt->loc.linnum);

    if (!stmt->asmcode)
        return;

    static std::string i_cns = "i";
    static std::string p_cns = "i";
    static std::string m_cns = "*m";
    static std::string mw_cns = "=*m";
    static std::string mrw_cns = "+*m";
    static std::string memory_name = "memory";

    AsmCode *code = static_cast<AsmCode *>(stmt->asmcode);
    std::vector<LLValue*> input_values;
    std::vector<std::string> input_constraints;
    std::vector<LLValue*> output_values;
    std::vector<std::string> output_constraints;
    std::vector<std::string> clobbers;

// FIXME
    //#define HOST_WIDE_INT long
    //HOST_WIDE_INT var_frame_offset; // "frame_offset" is a macro
    bool clobbers_mem = code->clobbersMemory;
    int input_idx = 0;
    int n_outputs = 0;
    int arg_map[10];

    assert(code->args.size() <= 10);

    std::vector<AsmArg>::iterator arg = code->args.begin();
    for (unsigned i = 0; i < code->args.size(); i++, ++arg) {
        bool is_input = true;
        LLValue* arg_val = 0;
        std::string cns;

        switch (arg->type) {
        case Arg_Integer:
            arg_val = toElem(arg->expr)->getRVal();
        do_integer:
            cns = i_cns;
            break;
        case Arg_Pointer:
            assert(arg->expr->op == TOKvar);
            arg_val = toElem(arg->expr)->getRVal();
            cns = p_cns;

            break;
        case Arg_Memory:
            arg_val = toElem(arg->expr)->getRVal();

            switch (arg->mode) {
            case Mode_Input:  cns = m_cns; break;
            case Mode_Output: cns = mw_cns;  is_input = false; break;
            case Mode_Update: cns = mrw_cns; is_input = false; break;
            default: llvm_unreachable("Unknown inline asm reference mode."); break;
            }
            break;
        case Arg_FrameRelative:
            // FIXME
            llvm_unreachable("Arg_FrameRelative not supported.");
/*          if (arg->expr->op == TOKvar)
                arg_val = ((VarExp *) arg->expr)->var->toSymbol()->Stree;
            else
                assert(0);
            if ( getFrameRelativeValue(arg_val, & var_frame_offset) ) {
//              arg_val = irs->integerConstant(var_frame_offset);
                cns = i_cns;
            } else {
                this->error("%s", "argument not frame relative");
                return;
            }
            if (arg->mode != Mode_Input)
                clobbers_mem = true;
            break;*/
        case Arg_LocalSize:
            // FIXME
            llvm_unreachable("Arg_LocalSize not supported.");
/*          var_frame_offset = cfun->x_frame_offset;
            if (var_frame_offset < 0)
                var_frame_offset = - var_frame_offset;
            arg_val = irs->integerConstant( var_frame_offset );*/
            goto do_integer;
        default:
            llvm_unreachable("Unknown inline asm reference type.");
        }

        if (is_input) {
            arg_map[i] = --input_idx;
            input_values.push_back(arg_val);
            input_constraints.push_back(cns);
        } else {
            arg_map[i] = n_outputs++;
            output_values.push_back(arg_val);
            output_constraints.push_back(cns);
        }
    }

    // Telling GCC that callee-saved registers are clobbered makes it preserve
    // those registers.   This changes the stack from what a naked function
    // expects.

// FIXME
//    if (! irs->func->naked) {
        assert(asmparser);
        for (size_t i = 0; i < code->regs.size(); i++) {
            if (code->regs[i]) {
                clobbers.push_back(asmparser->getRegName(i));
            }
        }
        if (clobbers_mem)
            clobbers.push_back(memory_name);
//    }

    // Remap argument numbers
    for (unsigned i = 0; i < code->args.size(); i++) {
        if (arg_map[i] < 0)
            arg_map[i] = -arg_map[i] - 1 + n_outputs;
    }

    bool pct = false;
    std::string::iterator
        p = code->insnTemplate.begin(),
        q = code->insnTemplate.end();
    //printf("start: %.*s\n", code->insnTemplateLen, code->insnTemplate);
    while (p < q) {
        if (pct) {
            if (*p >= '0' && *p <= '9') {
                // %% doesn't check against nargs
                *p = '0' + arg_map[*p - '0'];
                pct = false;
            } else if (*p == '$') {
                pct = false;
            }
            //assert(*p == '%');// could be 'a', etc. so forget it..
        } else if (*p == '$')
            pct = true;
        ++p;
    }

    typedef std::vector<std::string>::iterator It;
    IF_LOG {
        Logger::cout() << "final asm: " << code->insnTemplate << '\n';
        std::ostringstream ss;

        ss << "GCC-style output constraints: {";
        for (It i = output_constraints.begin(), e = output_constraints.end(); i != e; ++i) {
            ss << " " << *i;
        }
        ss << " }";
        Logger::println("%s", ss.str().c_str());

        ss.str("");
        ss << "GCC-style input constraints: {";
        for (It i = input_constraints.begin(), e = input_constraints.end(); i != e; ++i) {
            ss << " " << *i;
        }
        ss << " }";
        Logger::println("%s", ss.str().c_str());

        ss.str("");
        ss << "GCC-style clobbers: {";
        for (It i = clobbers.begin(), e = clobbers.end(); i != e; ++i) {
            ss << " " << *i;
        }
        ss << " }";
        Logger::println("%s", ss.str().c_str());
    }

    // rewrite GCC-style constraints to LLVM-style constraints
    std::string llvmOutConstraints;
    std::string llvmInConstraints;
    int n = 0;
    for(It i = output_constraints.begin(), e = output_constraints.end(); i != e; ++i, ++n) {
        // rewrite update constraint to in and out constraints
        if((*i)[0] == '+') {
            assert(*i == mrw_cns && "What else are we updating except memory?");
            /* LLVM doesn't support updating operands, so split into an input
             * and an output operand.
             */

            // Change update operand to pure output operand.
            *i = mw_cns;

            // Add input operand with same value, with original as "matching output".
            std::ostringstream ss;
            ss << '*' << (n + asmblock->outputcount);
            // Must be at the back; unused operands before used ones screw up numbering.
            input_constraints.push_back(ss.str());
            input_values.push_back(output_values[n]);
        }
        llvmOutConstraints += *i;
        llvmOutConstraints += ",";
    }
    asmblock->outputcount += n;

    for(It i = input_constraints.begin(), e = input_constraints.end(); i != e; ++i) {
        llvmInConstraints += *i;
        llvmInConstraints += ",";
    }

    std::string clobstr;
    for(It i = clobbers.begin(), e = clobbers.end(); i != e; ++i) {
        clobstr = "~{" + *i + "},";
        asmblock->clobs.insert(clobstr);
    }

    IF_LOG {
        typedef std::vector<LLValue*>::iterator It;
        {
            Logger::println("Output values:");
            LOG_SCOPE
            size_t i = 0;
            for (It I = output_values.begin(), E = output_values.end(); I != E; ++I) {
                Logger::cout() << "Out " << i++ << " = " << **I << '\n';
            }
        }
        {
            Logger::println("Input values:");
            LOG_SCOPE
            size_t i = 0;
            for (It I = input_values.begin(), E = input_values.end(); I != E; ++I) {
                Logger::cout() << "In  " << i++ << " = " << **I << '\n';
            }
        }
    }

    // excessive commas are removed later...

    replace_func_name(irs, code->insnTemplate);

    // push asm statement
    IRAsmStmt* asmStmt = new IRAsmStmt;
    asmStmt->code = code->insnTemplate;
    asmStmt->out_c = llvmOutConstraints;
    asmStmt->in_c = llvmInConstraints;
    asmStmt->out.insert(asmStmt->out.begin(), output_values.begin(), output_values.end());
    asmStmt->in.insert(asmStmt->in.begin(), input_values.begin(), input_values.end());
    asmStmt->isBranchToLabel = stmt->isBranchToLabel;
    asmblock->s.push_back(asmStmt);
}

//////////////////////////////////////////////////////////////////////////////

// rewrite argument indices to the block scope indices
static void remap_outargs(std::string& insnt, size_t nargs, size_t idx)
{
    static const std::string digits[10] =
    {
        "0","1","2","3","4",
        "5","6","7","8","9"
    };
    assert(nargs <= 10);

    static const std::string prefix("<<out");
    static const std::string suffix(">>");
    std::string argnum;
    std::string needle;
    char buf[10];
    for (unsigned i = 0; i < nargs; i++) {
        needle = prefix + digits[i] + suffix;
        size_t pos = insnt.find(needle);
        if(std::string::npos != pos)
            sprintf(buf, "%lu", idx++);
        while(std::string::npos != (pos = insnt.find(needle)))
            insnt.replace(pos, needle.size(), buf);
    }
}

// rewrite argument indices to the block scope indices
static void remap_inargs(std::string& insnt, size_t nargs, size_t idx)
{
    static const std::string digits[10] =
    {
        "0","1","2","3","4",
        "5","6","7","8","9"
    };
    assert(nargs <= 10);

    static const std::string prefix("<<in");
    static const std::string suffix(">>");
    std::string argnum;
    std::string needle;
    char buf[10];
    for (unsigned i = 0; i < nargs; i++) {
        needle = prefix + digits[i] + suffix;
        size_t pos = insnt.find(needle);
        if(std::string::npos != pos)
            sprintf(buf, "%lu", idx++);
        while(std::string::npos != (pos = insnt.find(needle)))
            insnt.replace(pos, needle.size(), buf);
    }
}

void CompoundAsmStatement_toIR(CompoundAsmStatement *stmt, IRState* p)
{
    IF_LOG Logger::println("AsmBlockStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    // disable inlining by default
    if (!p->func()->decl->allowInlining)
        p->func()->setNeverInline();

    // create asm block structure
    assert(!p->asmBlock);
    IRAsmBlock* asmblock = new IRAsmBlock(stmt);
    assert(asmblock);
    p->asmBlock = asmblock;

    // do asm statements
    for (unsigned i=0; i < stmt->statements->dim; i++)
    {
        Statement* s = (*stmt->statements)[i];
        if (s) {
            Statement_toIR(s, p);
        }
    }

    // build forwarder for in-asm branches to external labels
    // this additional asm code sets the __llvm_jump_target variable
    // to a unique value that will identify the jump target in
    // a post-asm switch

    // maps each goto destination to its special value
    std::map<LabelDsymbol*, int> gotoToVal;

    // location of the special value determining the goto label
    // will be set if post-asm dispatcher block is needed
    llvm::AllocaInst* jump_target = 0;

    {
        FuncDeclaration* fd = gIR->func()->decl;
        const char* fdmangle = mangle(fd);

        // we use a simple static counter to make sure the new end labels are unique
        static size_t uniqueLabelsId = 0;
        std::ostringstream asmGotoEndLabel;
        printLabelName(asmGotoEndLabel, fdmangle, "_llvm_asm_end");
        asmGotoEndLabel << uniqueLabelsId++;

        // initialize the setter statement we're going to build
        IRAsmStmt* outSetterStmt = new IRAsmStmt;
        std::string asmGotoEnd = "\n\tjmp "+asmGotoEndLabel.str()+"\n";
        std::ostringstream code;
        code << asmGotoEnd;

        int n_goto = 1;

        size_t n = asmblock->s.size();
        for(size_t i=0; i<n; ++i)
        {
            IRAsmStmt* a = asmblock->s[i];

            // skip non-branch statements
            if(!a->isBranchToLabel)
                continue;

            // if internal, no special handling is necessary, skip
            std::vector<Identifier*>::const_iterator it, end;
            end = asmblock->internalLabels.end();
            bool skip = false;
            for(it = asmblock->internalLabels.begin(); it != end; ++it)
                if((*it)->equals(a->isBranchToLabel->ident))
                    skip = true;
            if(skip)
                continue;

            // if we already set things up for this branch target, skip
            if(gotoToVal.find(a->isBranchToLabel) != gotoToVal.end())
                continue;

            // record that the jump needs to be handled in the post-asm dispatcher
            gotoToVal[a->isBranchToLabel] = n_goto;

            // provide an in-asm target for the branch and set value
            IF_LOG Logger::println("statement '%s' references outer label '%s': creating forwarder", a->code.c_str(), a->isBranchToLabel->ident->string);
            printLabelName(code, fdmangle, a->isBranchToLabel->ident->string);
            code << ":\n\t";
            code << "movl $<<in" << n_goto << ">>, $<<out0>>\n";
            //FIXME: Store the value -> label mapping somewhere, so it can be referenced later
            outSetterStmt->in.push_back(DtoConstUint(n_goto));
            outSetterStmt->in_c += "i,";
            code << asmGotoEnd;

            ++n_goto;
        }
        if(code.str() != asmGotoEnd)
        {
            // finalize code
            outSetterStmt->code = code.str();
            outSetterStmt->code += asmGotoEndLabel.str()+":\n";

            // create storage for and initialize the temporary
            jump_target = DtoAlloca(Type::tint32, "__llvm_jump_target");
            gIR->ir->CreateStore(DtoConstUint(0), jump_target);
            // setup variable for output from asm
            outSetterStmt->out_c = "=*m,";
            outSetterStmt->out.push_back(jump_target);

            asmblock->s.push_back(outSetterStmt);
        }
        else
            delete outSetterStmt;
    }


    // build a fall-off-end-properly asm statement

    FuncDeclaration* thisfunc = p->func()->decl;
    bool useabiret = false;
    p->asmBlock->asmBlock->abiret = NULL;
    if (thisfunc->fbody->endsWithAsm() == stmt && thisfunc->type->nextOf()->ty != Tvoid)
    {
        // there can't be goto forwarders in this case
        assert(gotoToVal.empty());
        emitABIReturnAsmStmt(asmblock, stmt->loc, thisfunc);
        useabiret = true;
    }


    // build asm block
    std::vector<LLValue*> outargs;
    std::vector<LLValue*> inargs;
    std::vector<LLType*> outtypes;
    std::vector<LLType*> intypes;
    std::string out_c;
    std::string in_c;
    std::string clobbers;
    std::string code;
    size_t asmIdx = asmblock->retn;

    Logger::println("do outputs");
    size_t n = asmblock->s.size();
    for (size_t i=0; i<n; ++i)
    {
        IRAsmStmt* a = asmblock->s[i];
        assert(a);
        size_t onn = a->out.size();
        for (size_t j=0; j<onn; ++j)
        {
            outargs.push_back(a->out[j]);
            outtypes.push_back(a->out[j]->getType());
        }
        if (!a->out_c.empty())
        {
            out_c += a->out_c;
        }
        remap_outargs(a->code, onn+a->in.size(), asmIdx);
        asmIdx += onn;
    }

    Logger::println("do inputs");
    for (size_t i=0; i<n; ++i)
    {
        IRAsmStmt* a = asmblock->s[i];
        assert(a);
        size_t inn = a->in.size();
        for (size_t j=0; j<inn; ++j)
        {
            inargs.push_back(a->in[j]);
            intypes.push_back(a->in[j]->getType());
        }
        if (!a->in_c.empty())
        {
            in_c += a->in_c;
        }
        remap_inargs(a->code, inn+a->out.size(), asmIdx);
        asmIdx += inn;
        if (!code.empty())
            code += "\n\t";
        code += a->code;
    }
    asmblock->s.clear();

    // append inputs
    out_c += in_c;

    // append clobbers
    typedef std::set<std::string>::iterator clobs_it;
    for (clobs_it i=asmblock->clobs.begin(); i!=asmblock->clobs.end(); ++i)
    {
        out_c += *i;
    }

    // remove excessive comma
    if (!out_c.empty())
        out_c.resize(out_c.size()-1);

    IF_LOG {
        Logger::println("code = \"%s\"", code.c_str());
        Logger::println("constraints = \"%s\"", out_c.c_str());
    }

    // build return types
    LLType* retty;
    if (asmblock->retn)
        retty = asmblock->retty;
    else
        retty = llvm::Type::getVoidTy(gIR->context());

    // build argument types
    std::vector<LLType*> types;
    types.insert(types.end(), outtypes.begin(), outtypes.end());
    types.insert(types.end(), intypes.begin(), intypes.end());
    llvm::FunctionType* fty = llvm::FunctionType::get(retty, types, false);
    IF_LOG Logger::cout() << "function type = " << *fty << '\n';

    std::vector<LLValue*> args;
    args.insert(args.end(), outargs.begin(), outargs.end());
    args.insert(args.end(), inargs.begin(), inargs.end());

    IF_LOG {
        Logger::cout() << "Arguments:" << '\n';
        Logger::indent();
        for (std::vector<LLValue*>::iterator b = args.begin(), i = b, e = args.end(); i != e; ++i) {
            Stream cout = Logger::cout();
            cout << '$' << (i - b) << " ==> " << **i;
            if (!llvm::isa<llvm::Instruction>(*i) && !llvm::isa<LLGlobalValue>(*i))
                cout << '\n';
        }
        Logger::undent();
    }

    llvm::InlineAsm* ia = llvm::InlineAsm::get(fty, code, out_c, true);

    llvm::CallInst* call = p->ir->CreateCall(ia, args,
        retty == LLType::getVoidTy(gIR->context()) ? "" : "asm");

    IF_LOG Logger::cout() << "Complete asm statement: " << *call << '\n';

    // capture abi return value
    if (useabiret)
    {
        IRAsmBlock* block = p->asmBlock;
        if (block->retfixup)
            block->asmBlock->abiret = (*block->retfixup)(p->ir, call);
        else if (p->asmBlock->retemu)
            block->asmBlock->abiret = DtoLoad(block->asmBlock->abiret);
        else
            block->asmBlock->abiret = call;
    }

    p->asmBlock = NULL;

    // if asm contained external branches, emit goto forwarder code
    if(!gotoToVal.empty())
    {
        assert(jump_target);

        // make new blocks
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterasmgotoforwarder", p->topfunc(), oldend);

        llvm::LoadInst* val = p->ir->CreateLoad(jump_target, "__llvm_jump_target_value");
        llvm::SwitchInst* sw = p->ir->CreateSwitch(val, bb, gotoToVal.size());

        // add all cases
        std::map<LabelDsymbol*, int>::iterator it, end = gotoToVal.end();
        for(it = gotoToVal.begin(); it != end; ++it)
        {
            llvm::BasicBlock* casebb = llvm::BasicBlock::Create(gIR->context(), "case", p->topfunc(), bb);
            sw->addCase(LLConstantInt::get(llvm::IntegerType::get(gIR->context(), 32), it->second), casebb);

            p->scope() = IRScope(casebb,bb);
            DtoGoto(stmt->loc, it->first, stmt->enclosingFinally);
        }

        p->scope() = IRScope(bb,oldend);
    }
}

//////////////////////////////////////////////////////////////////////////////

CompoundAsmStatement* Statement::endsWithAsm()
{
    // does not end with inline asm
    return NULL;
}

CompoundAsmStatement* CompoundStatement::endsWithAsm()
{
    // make the last inner statement decide
    if (statements && statements->dim)
    {
        unsigned last = statements->dim - 1;
        Statement* s = (*statements)[last];
        if (s) return s->endsWithAsm();
    }
    return NULL;
}

CompoundAsmStatement* CompoundAsmStatement::endsWithAsm()
{
    // yes this is inline asm
    return this;
}

//////////////////////////////////////////////////////////////////////////////

void AsmStatement_toNakedIR(AsmStatement *stmt, IRState *irs)
{
    IF_LOG Logger::println("AsmStatement::toNakedIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    // is there code?
    if (!stmt->asmcode)
        return;
    AsmCode * code = static_cast<AsmCode *>(stmt->asmcode);

    // build asm stmt
    replace_func_name(irs, code->insnTemplate);
    irs->nakedAsm << "\t" << code->insnTemplate << std::endl;
}
