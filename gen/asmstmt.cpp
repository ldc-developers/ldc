//===-- asmstmt.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file originates from work by David Friedman for GDC released under
// the GPL 2 and Artistic licenses. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/scope.h"
#include "dmd/statement.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "llvm/IR/InlineAsm.h"
#include <cassert>
#include <cstring>
#include <deque>
#include <string>
#include <sstream>

typedef enum {
  Arg_Integer,
  Arg_Pointer,
  Arg_Memory,
  Arg_FrameRelative,
  Arg_LocalSize,
  Arg_Dollar
} AsmArgType;

typedef enum { Mode_Input, Mode_Output, Mode_Update } AsmArgMode;

struct AsmArg {
  Expression *expr;
  AsmArgType type;
  AsmArgMode mode;
  AsmArg(AsmArgType type, Expression *expr, AsmArgMode mode) {
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
  int clobbersMemory;
  explicit AsmCode(int n_regs) {
    regs.resize(n_regs, false);
    dollarLabel = 0;
    clobbersMemory = 0;
  }
};

struct AsmParserCommon {
  virtual ~AsmParserCommon() = default;
  virtual void run(Scope *sc, InlineAsmStatement *asmst) = 0;
  virtual std::string getRegName(int i) = 0;
};
AsmParserCommon *asmparser = nullptr;

#include "asm-x86.h" // x86 assembly parser
#define ASM_X86_64
#include "asm-x86.h" // x86_64 assembly parser
#undef ASM_X86_64

using namespace dmd;

/**
 * Replaces <<func>> with the name of the currently codegen'd function.
 *
 * This kludge is required to handle labels correctly, as the instruction
 * strings for jumps, … are generated during semantic3, but attribute inference
 * might change the function type (and hence the mangled name) right at the end
 * of semantic3.
 */
static void replace_func_name(IRState *p, std::string &insnt) {
  static const std::string needle("<<func>>");

  const char *mangle = mangleExact(p->func()->decl);

  size_t pos;
  while (std::string::npos != (pos = insnt.find(needle))) {
    // This will only happen for few instructions, and only once for those.
    insnt.replace(pos, needle.size(), mangle);
  }
}

Statement *asmSemantic(AsmStatement *s, Scope *sc) {
  if (!s->tokens) {
    return nullptr;
  }

  sc->func->hasReturnExp |= 8;

  // GCC-style asm starts with a string literal or a `(`
  if (s->tokens->value == TOK::string_ ||
      s->tokens->value == TOK::leftParenthesis) {
    auto gas = createGccAsmStatement(s->loc, s->tokens);
    return gccAsmSemantic(gas, sc);
  }

  // this is DMD-style asm
  sc->func->hasReturnExp |= 32;

  const auto caseSensitive = s->caseSensitive;

  auto ias = createInlineAsmStatement(s->loc, s->tokens);
  s = ias;
  s->caseSensitive = caseSensitive;

  bool err = false;
  llvm::Triple const &t = *global.params.targetTriple;
  if (!(t.getArch() == llvm::Triple::x86 ||
        t.getArch() == llvm::Triple::x86_64)) {
    error(s->loc,
        "DMD-style `asm { op; }` statements are not supported for the \"%s\" "
        "architecture.",
        t.getArchName().str().c_str());
    errorSupplemental(s->loc, "Use GDC-style `asm { \"op\" : …; }` syntax or "
                              "`ldc.llvmasm.__asm` instead.");
    err = true;
  }
  if (!global.params.useInlineAsm) {
    error(s->loc,
        "the `asm` statement is not allowed when the -noasm switch is used");
    err = true;
  }
  if (err) {
    if (!global.gag) {
      fatal();
    }
    return s;
  }

  // puts(toChars());

  if (!asmparser) {
    if (t.getArch() == llvm::Triple::x86) {
      asmparser = new AsmParserx8632::AsmParser;
    } else if (t.getArch() == llvm::Triple::x86_64) {
      asmparser = new AsmParserx8664::AsmParser;
    }
  }

  asmparser->run(sc, ias);

  return s;
}

void AsmStatement_toIR(InlineAsmStatement *stmt, IRState *irs) {
  IF_LOG Logger::println("InlineAsmStatement::toIR(): %s", stmt->loc.toChars());
  LOG_SCOPE;

  // sanity check
  assert((irs->func()->decl->hasReturnExp & 40) == 40);

  // get asm block
  IRAsmBlock *asmblock = irs->asmBlock;
  assert(asmblock);

  // debug info
  gIR->DBuilder.EmitStopPoint(stmt->loc);

  if (!stmt->asmcode) {
    return;
  }

  static std::string i_cns = "i";
  static std::string p_cns = "i";
  static std::string m_cns = "*m";
  static std::string mw_cns = "=*m";
  static std::string mrw_cns = "+*m";
  static std::string memory_name = "memory";

  AsmCode *code = static_cast<AsmCode *>(stmt->asmcode);
  auto asmStmt = new IRAsmStmt;
  asmStmt->isBranchToLabel = stmt->isBranchToLabel;

  std::vector<std::string> input_constraints;
  std::vector<std::string> output_constraints;
  std::vector<std::string> clobbers;

  // FIXME
  //#define HOST_WIDE_INT long
  // HOST_WIDE_INT var_frame_offset; // "frame_offset" is a macro
  bool clobbers_mem = code->clobbersMemory;
  int input_idx = 0;
  int n_outputs = 0;
  int arg_map[10];

  assert(code->args.size() <= 10);

  auto arg = code->args.begin();
  for (unsigned i = 0; i < code->args.size(); i++, ++arg) {
    bool is_input = true;
    LLValue *arg_val = nullptr;
    std::string cns;

    switch (arg->type) {
    case Arg_Integer:
      arg_val = DtoRVal(arg->expr);
    do_integer:
      cns = i_cns;
      break;
    case Arg_Pointer:
      assert(arg->expr->isVarExp());
      arg_val = DtoRVal(arg->expr);
      cns = p_cns;

      break;
    case Arg_Memory:
      arg_val = DtoRVal(arg->expr);

      switch (arg->mode) {
      case Mode_Input:
        cns = m_cns;
        break;
      case Mode_Output:
        cns = mw_cns;
        is_input = false;
        break;
      case Mode_Update:
        cns = mrw_cns;
        is_input = false;
        break;
      }
      break;
    case Arg_FrameRelative:
      // FIXME
      llvm_unreachable("Arg_FrameRelative not supported.");
    /*          if (auto ve = arg->expr->isVarExp())
                    arg_val = ve->var->toSymbol()->Stree;
                else
                    assert(0);
                if ( getFrameRelativeValue(arg_val, & var_frame_offset) ) {
    //              arg_val = irs->integerConstant(var_frame_offset);
                    cns = i_cns;
                } else {
                    error(stmt->loc, "argument not frame relative");
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
      asmStmt->in.ops.push_back(arg_val);
      input_constraints.push_back(cns);
      asmStmt->in.dTypes.push_back(arg->expr->type);
    } else {
      arg_map[i] = n_outputs++;
      asmStmt->out.ops.push_back(arg_val);
      output_constraints.push_back(cns);
      asmStmt->out.dTypes.push_back(arg->expr->type);
    }
  }

  // Telling GCC that callee-saved registers are clobbered makes it preserve
  // those registers.   This changes the stack from what a naked function
  // expects.

  // FIXME
  //    if (!irs->func->isNaked()) {
  assert(asmparser);
  for (size_t i = 0; i < code->regs.size(); i++) {
    if (code->regs[i]) {
      clobbers.push_back(asmparser->getRegName(i));
    }
  }
  if (clobbers_mem) {
    clobbers.push_back(memory_name);
  }
  //    }

  // Remap argument numbers
  for (unsigned i = 0; i < code->args.size(); i++) {
    if (arg_map[i] < 0) {
      arg_map[i] = -arg_map[i] - 1 + n_outputs;
    }
  }

  bool pct = false;
  auto p = code->insnTemplate.begin();
  auto q = code->insnTemplate.end();
  // printf("start: %.*s\n", code->insnTemplateLen, code->insnTemplate);
  while (p < q) {
    if (pct) {
      if (*p >= '0' && *p <= '9') {
        // %% doesn't check against nargs
        *p = '0' + arg_map[*p - '0'];
        pct = false;
      } else if (*p == '$') {
        pct = false;
      }
      // assert(*p == '%');// could be 'a', etc. so forget it..
    } else if (*p == '$') {
      pct = true;
    }
    ++p;
  }

  IF_LOG {
    Logger::cout() << "final asm: " << code->insnTemplate << '\n';
    std::ostringstream ss;

    ss << "GCC-style output constraints: {";
    for (const auto &oc : output_constraints) {
      ss << " " << oc;
    }
    ss << " }";
    Logger::println("%s", ss.str().c_str());

    ss.str("");
    ss << "GCC-style input constraints: {";
    for (const auto &ic : input_constraints) {
      ss << " " << ic;
    }
    ss << " }";
    Logger::println("%s", ss.str().c_str());

    ss.str("");
    ss << "GCC-style clobbers: {";
    for (const auto &c : clobbers) {
      ss << " " << c;
    }
    ss << " }";
    Logger::println("%s", ss.str().c_str());
  }

  // rewrite GCC-style constraints to LLVM-style constraints
  int n = 0;
  for (auto &oc : output_constraints) {
    // rewrite update constraint to in and out constraints
    if (oc[0] == '+') {
      assert(oc == mrw_cns && "What else are we updating except memory?");
      /* LLVM doesn't support updating operands, so split into an input
       * and an output operand.
       */

      // Change update operand to pure output operand.
      oc = mw_cns;

      // Add input operand with same value, with original as "matching
      // output".
      std::ostringstream ss;
      ss << '*' << (n + asmblock->outputcount);
      // Must be at the back; unused operands before used ones screw up
      // numbering.
      input_constraints.push_back(ss.str());
      asmStmt->in.ops.push_back(asmStmt->out.ops[n]);
      asmStmt->in.dTypes.push_back(asmStmt->out.dTypes[n]);
    }
    asmStmt->out.c += oc;
    asmStmt->out.c += ",";
    n++;
  }
  asmblock->outputcount += n;

  for (const auto &ic : input_constraints) {
    asmStmt->in.c += ic;
    asmStmt->in.c += ",";
  }

  std::string clobstr;
  for (const auto &c : clobbers) {
    clobstr = "~{" + c + "},";
    asmblock->clobs.insert(clobstr);
  }

  IF_LOG {
    {
      Logger::println("Output values:");
      LOG_SCOPE
      size_t i = 0;
      for (auto ov : asmStmt->out.ops) {
        Logger::cout() << "Out " << i++ << " = " << *ov << '\n';
      }
    }
    {
      Logger::println("Input values:");
      LOG_SCOPE
      size_t i = 0;
      for (auto iv : asmStmt->in.ops) {
        Logger::cout() << "In  " << i++ << " = " << *iv << '\n';
      }
    }
  }

  // excessive commas are removed later...

  replace_func_name(irs, code->insnTemplate);

  // push asm statement

  asmStmt->code = code->insnTemplate;
  asmblock->s.push_back(asmStmt);
}

//////////////////////////////////////////////////////////////////////////////

// rewrite argument indices to the block scope indices
static void remap_args(std::string &insnt, size_t nargs, size_t idx,
                       const std::string& prefix) {
  static const std::string digits[10] = {"0", "1", "2", "3", "4",
                                         "5", "6", "7", "8", "9"};
  assert(nargs <= 10);

  static const std::string suffix(">>");
  std::string argnum;
  std::string needle;
  char buf[10];
  for (unsigned i = 0; i < nargs; i++) {
    needle = prefix + digits[i] + suffix;
    size_t pos = insnt.find(needle);
    if (std::string::npos != pos) {
      snprintf(buf, 10, "%llu", static_cast<unsigned long long>(idx++));
    }
    while (std::string::npos != (pos = insnt.find(needle))) {
      insnt.replace(pos, needle.size(), buf);
    }
  }
}

void CompoundAsmStatement_toIR(CompoundAsmStatement *stmt, IRState *p) {
  IF_LOG Logger::println("CompoundAsmStatement::toIR(): %s",
                         stmt->loc.toChars());
  LOG_SCOPE;

  const bool isCompoundGccAsmStatement =
      (stmt->statements && stmt->statements->length &&
       stmt->statements->front()->isGccAsmStatement());
  if (isCompoundGccAsmStatement) {
    for (Statement *s : *stmt->statements) {
      if (auto gas = s->isGccAsmStatement()) {
        Statement_toIR(gas, p);
      } else {
        error(s->loc,
              "DMD-style assembly statement unsupported within GCC-style "
              "`asm` block");
        fatal();
      }
    }

    return;
  }

  // disable inlining by default
  if (!p->func()->decl->allowInlining) {
    p->func()->setNeverInline();
  }

  // create asm block structure
  assert(!p->asmBlock);
  auto asmblock = new IRAsmBlock(stmt);
  assert(asmblock);
  p->asmBlock = asmblock;

  // do asm statements
  for (Statement *s : *stmt->statements) {
    if (s) {
      if (s->isGccAsmStatement()) {
        error(s->loc,
              "GCC-style assembly statement unsupported within DMD-style "
              "`asm` block");
        fatal();
      }
      Statement_toIR(s, p);
    }
  }

  // build forwarder for in-asm branches to external labels
  // this additional asm code sets the __llvm_jump_target variable
  // to a unique value that will identify the jump target in
  // a post-asm switch

  // maps each goto destination to its special value
  std::map<LabelDsymbol *, int> gotoToVal;

  // location of the special value determining the goto label
  // will be set if post-asm dispatcher block is needed
  LLValue *jump_target = nullptr;

  {
    FuncDeclaration *fd = gIR->func()->decl;
    const char *fdmangle = mangleExact(fd);

    // we use a simple static counter to make sure the new end labels are
    // unique
    static size_t uniqueLabelsId = 0;
    std::ostringstream asmGotoEndLabel;
    printLabelName(asmGotoEndLabel, fdmangle, "_llvm_asm_end");
    asmGotoEndLabel << uniqueLabelsId++;

    // initialize the setter statement we're going to build
    auto outSetterStmt = new IRAsmStmt;
    std::string asmGotoEnd = "\n\tjmp " + asmGotoEndLabel.str() + "\n";
    std::ostringstream code;
    code << asmGotoEnd;

    int n_goto = 1;

    for (IRAsmStmt *a : asmblock->s) {
      // skip non-branch statements
      LabelDsymbol *const targetLabel = a->isBranchToLabel;
      if (!targetLabel) {
        continue;
      }
      Identifier *const ident = targetLabel->ident;

      // if internal, no special handling is necessary, skip
      if (llvm::any_of(asmblock->internalLabels,
                       [ident](Identifier *i) { return i->equals(ident); })) {
        continue;
      }

      // if we already set things up for this branch target, skip
      if (gotoToVal.find(targetLabel) != gotoToVal.end()) {
        continue;
      }

      // record that the jump needs to be handled in the post-asm dispatcher
      gotoToVal[targetLabel] = n_goto;

      // provide an in-asm target for the branch and set value
      IF_LOG Logger::println(
          "statement '%s' references outer label '%s': creating forwarder",
          a->code.c_str(), ident->toChars());
      printLabelName(code, fdmangle, ident->toChars());
      code << ":\n\t";
      code << "movl $<<in" << n_goto << ">>, $<<out0>>\n";
      // FIXME: Store the value -> label mapping somewhere, so it can be
      // referenced later
      outSetterStmt->in.ops.push_back(DtoConstUint(n_goto));
      outSetterStmt->in.dTypes.push_back(Type::tuns32);
      outSetterStmt->in.c += "i,";
      code << asmGotoEnd;

      ++n_goto;
    }
    if (code.str() != asmGotoEnd) {
      // finalize code
      outSetterStmt->code = code.str();
      outSetterStmt->code += asmGotoEndLabel.str() + ":\n";

      // create storage for and initialize the temporary
      jump_target = DtoAllocaDump(DtoConstUint(0), 0, "__llvm_jump_target");
      // setup variable for output from asm
      outSetterStmt->out.c = "=*m,";
      outSetterStmt->out.ops.push_back(jump_target);
      outSetterStmt->out.dTypes.push_back(Type::tuns32);

      asmblock->s.push_back(outSetterStmt);
    } else {
      delete outSetterStmt;
    }
  }

  // build a fall-off-end-properly asm statement

  FuncDeclaration *thisfunc = p->func()->decl;
  bool useabiret = false;
  p->asmBlock->asmBlock->abiret = nullptr;
  if (thisfunc->fbody->endsWithAsm() == stmt &&
      thisfunc->type->nextOf()->ty != TY::Tvoid) {
    // there can't be goto forwarders in this case
    assert(gotoToVal.empty());
    emitABIReturnAsmStmt(asmblock, stmt->loc, thisfunc);
    useabiret = true;
  }

  // build asm block
  struct ArgBlock {
    ArgBlock() = default;
    std::vector<LLValue *> args;
    std::vector<LLType *> types;
    std::vector<Type *> dTypes;
    std::string c;
  } in, out;
  std::string clobbers;
  std::string code;
  size_t asmIdx = asmblock->retn;

  Logger::println("do outputs");
  size_t n = asmblock->s.size();
  for (size_t i = 0; i < n; ++i) {
    IRAsmStmt *a = asmblock->s[i];
    assert(a);
    size_t onn = a->out.ops.size();
    for (size_t j = 0; j < onn; ++j) {
      out.args.push_back(a->out.ops[j]);
      out.types.push_back(a->out.ops[j]->getType());
      out.dTypes.push_back(a->out.dTypes[j]);
    }
    if (!a->out.c.empty()) {
      out.c += a->out.c;
    }
    remap_args(a->code, onn + a->in.ops.size(), asmIdx, "<<out");
    asmIdx += onn;
  }

  Logger::println("do inputs");
  for (size_t i = 0; i < n; ++i) {
    IRAsmStmt *a = asmblock->s[i];
    assert(a);
    size_t inn = a->in.ops.size();
    for (size_t j = 0; j < inn; ++j) {
      in.args.push_back(a->in.ops[j]);
      in.types.push_back(a->in.ops[j]->getType());
      in.dTypes.push_back(a->in.dTypes[j]);
    }
    if (!a->in.c.empty()) {
      in.c += a->in.c;
    }
    remap_args(a->code, inn + a->out.ops.size(), asmIdx, "<<in");
    asmIdx += inn;
    if (!code.empty()) {
      code += "\n\t";
    }
    code += a->code;
  }
  asmblock->s.clear();

  // append inputs
  out.c += in.c;

  // append clobbers
  for (const auto &c : asmblock->clobs) {
    out.c += c;
  }

  // remove excessive comma
  if (!out.c.empty()) {
    out.c.resize(out.c.size() - 1);
  }

  IF_LOG {
    Logger::println("code = \"%s\"", code.c_str());
    Logger::println("constraints = \"%s\"", out.c.c_str());
  }

  // build return types
  LLType *retty;
  if (asmblock->retn) {
    retty = asmblock->retty;
  } else {
    retty = llvm::Type::getVoidTy(gIR->context());
  }

  // build argument types
  std::vector<LLType *> types;
  types.insert(types.end(), out.types.begin(), out.types.end());
  types.insert(types.end(), in.types.begin(), in.types.end());
  llvm::FunctionType *fty = llvm::FunctionType::get(retty, types, false);
  IF_LOG Logger::cout() << "function type = " << *fty << '\n';

  std::vector<LLValue *> args;
  args.insert(args.end(), out.args.begin(), out.args.end());
  args.insert(args.end(), in.args.begin(), in.args.end());
    
  auto constraintInfo = llvm::InlineAsm::ParseConstraints(out.c);
  assert(constraintInfo.size() >= out.dTypes.size() + in.dTypes.size());
  std::vector<LLType *> indirectTypes;
  indirectTypes.reserve(out.dTypes.size() + in.dTypes.size());
  size_t i = asmblock->retn;
    
  for (Type *t : out.dTypes) {
    assert(constraintInfo[i].Type == llvm::InlineAsm::ConstraintPrefix::isOutput);
    if (constraintInfo[i].isIndirect) {
      if (TypePointer *pt = t->isTypePointer())
        indirectTypes.push_back(DtoMemType(pt->nextOf()));
      else
        indirectTypes.push_back(DtoMemType(t));
    }
    i++;
  }
  for (Type *t : in.dTypes) {
    assert(constraintInfo[i].Type == llvm::InlineAsm::ConstraintPrefix::isInput);
    if (constraintInfo[i].isIndirect) {
      if (TypePointer *pt = t->isTypePointer())
        indirectTypes.push_back(DtoType(pt->nextOf()));
      else
        indirectTypes.push_back(DtoType(t));
    }
    i++;
  }

  IF_LOG {
    Logger::cout() << "Arguments:" << '\n';
    Logger::indent();
    size_t i = 0;
    for (auto arg : args) {
      Stream cout = Logger::cout();
      cout << '$' << i << " ==> " << *arg;
      if (!llvm::isa<llvm::Instruction>(arg) &&
          !llvm::isa<LLGlobalValue>(arg)) {
        cout << '\n';
      }
      ++i;
    }
    Logger::undent();
  }

  for (; i < constraintInfo.size(); i++) {
    if (!constraintInfo[i].isIndirect)
      continue;
    llvm::errs() << "unhandled indirect constraint in" << out.c << "\nindex i = " << i << '\n';
    llvm::errs() << "function type = " << *fty << '\n';
    for (size_t j = 0; j < indirectTypes.size(); j++) {
      llvm::errs() << " " << *(indirectTypes[j]) << '\n';
    }
          
    llvm_unreachable("unhandled indirect constraint");
  }

  llvm::InlineAsm *ia = llvm::InlineAsm::get(fty, code, out.c, true);

  auto call = p->createInlineAsmCall(stmt->loc, ia, args, indirectTypes);
  if (!retty->isVoidTy()) {
    call->setName("asm");
  }

  IF_LOG Logger::cout() << "Complete asm statement: " << *call << '\n';

  // capture abi return value
  if (useabiret) {
    IRAsmBlock *block = p->asmBlock;
    if (block->retfixup) {
      block->asmBlock->abiret = (*block->retfixup)(p->ir, call);
    } else if (p->asmBlock->retemu) {
      block->asmBlock->abiret = DtoLoad(block->retty, block->asmBlock->abiret);
    } else {
      block->asmBlock->abiret = call;
    }
  }

  p->asmBlock = nullptr;

  // if asm contained external branches, emit goto forwarder code
  if (!gotoToVal.empty()) {
    assert(jump_target);

    // make new blocks
    llvm::BasicBlock *bb = p->insertBB("afterasmgotoforwarder");

    auto val = DtoLoad(LLType::getInt32Ty(gIR->context()), jump_target, "__llvm_jump_target_value");
    llvm::SwitchInst *sw = p->ir->CreateSwitch(val, bb, gotoToVal.size());

    // add all cases
    for (const auto &pair : gotoToVal) {
      llvm::BasicBlock *casebb = p->insertBBBefore(bb, "case");
      sw->addCase(LLConstantInt::get(llvm::IntegerType::get(gIR->context(), 32),
                                     pair.second),
                  casebb);

      p->ir->SetInsertPoint(casebb);
      DtoGoto(stmt->loc, pair.first);
    }

    p->ir->SetInsertPoint(bb);
  }
}

//////////////////////////////////////////////////////////////////////////////

void AsmStatement_toNakedIR(InlineAsmStatement *stmt, IRState *irs) {
  IF_LOG Logger::println("InlineAsmStatement::toNakedIR(): %s",
                         stmt->loc.toChars());
  LOG_SCOPE;

  // is there code?
  if (!stmt->asmcode) {
    return;
  }
  AsmCode *code = static_cast<AsmCode *>(stmt->asmcode);

  // build asm stmt
  replace_func_name(irs, code->insnTemplate);
  irs->nakedAsm << "\t" << code->insnTemplate << std::endl;
}
