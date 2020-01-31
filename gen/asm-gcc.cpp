//===-- asm-gcc.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// Converts a GDC/GCC-style inline assembly statement to an LLVM inline
// assembler expression.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/statement.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"

namespace {
llvm::StringRef peekString(StringExp *se) {
  DString slice = se->peekString();
  return {slice.ptr, slice.length};
}

class ConstraintsBuilder {
  std::ostringstream str; // LLVM constraints string being built
  LLSmallVector<bool, 8> isIndirectOutput;

  // Appends a constraint string expression with an optional prefix.
  // Returns true if the string describes an indirect operand.
  bool append(Expression *e, char prefix = 0) {
    auto se = e->isStringExp();
    assert(se);
    llvm::StringRef code = peekString(se);
    assert(!code.empty());

    // commit prefix and strip from `code`, if present
    if (prefix) {
      str << prefix;
      if (code[0] == prefix)
        code = code.substr(1);
    }

    // commit any modifier and strip from `code`
    bool isIndirect = false;
    if (code.startswith("&")) { // early clobber
      str << '&';
      code = code.substr(1);
    } else if (code.startswith("*")) { // indirect in/output
      isIndirect = true;
      str << '*';
      code = code.substr(1);
    }

    bool withCurlyBraces = needsCurlyBraces(code);

    if (withCurlyBraces)
      str << '{';
    str.write(code.data(), code.size());
    if (withCurlyBraces)
      str << '}';

    str << ',';

    return isIndirect;
  }

  // Register names need to be enclosed in curly braces for LLVM.
  bool needsCurlyBraces(llvm::StringRef code) {
    auto len = code.size();
    bool noCurlyBraces = (len == 1 || (len == 3 && code.startswith("^")) ||
                          code.startswith("{"));
    return !noCurlyBraces;
  }

public:
  // Returns the final constraints string for LLVM for a GCC-style asm
  // statement.
  std::string build(GccAsmStatement *stmt) {
    str.clear();
    isIndirectOutput.clear();
    isIndirectOutput.resize(stmt->outputargs, false);

    if (stmt->constraints) {
      for (size_t i = 0; i < stmt->constraints->length; ++i) {
        bool isOutput = (i < stmt->outputargs);
        bool isIndirect = append((*stmt->constraints)[i], isOutput ? '=' : 0);
        if (isOutput && isIndirect)
          isIndirectOutput[i] = true;
      }
    }

    if (stmt->clobbers) {
      for (auto e : *stmt->clobbers) {
        append(e, '~');
      }
    }

    // remove excessive comma
    std::string result = str.str();
    if (auto size = result.size())
      result.resize(size - 1);

    return result;
  }

  bool isIndirectOutputOperand(size_t outputOperandIndex) const {
    return isIndirectOutput[outputOperandIndex];
  }
};
}

void GccAsmStatement_toIR(GccAsmStatement *stmt, IRState *irs) {
  IF_LOG Logger::println("GccAsmStatement::toIR(): %s", stmt->loc.toChars());
  LOG_SCOPE;

  if (stmt->labels) {
    stmt->error("labels in GCC-style assembly are not supported yet");
    fatal();
  }

  llvm::StringRef insn = peekString(stmt->insn->isStringExp());

  ConstraintsBuilder constraintsBuilder;
  const std::string constraints = constraintsBuilder.build(stmt);

  LLSmallVector<LLValue *, 8> outputLVals;
  LLSmallVector<LLType *, 8> outputTypes;
  LLSmallVector<LLValue *, 8> indirectOutputLVals;
  LLSmallVector<Expression *, 8> inputArgs;
  if (stmt->args) {
    for (size_t i = 0; i < stmt->args->length; ++i) {
      Expression *e = (*stmt->args)[i];
      const bool isOutput = (i < stmt->outputargs);
      if (isOutput) {
        LLValue *lval = DtoLVal(e);
        if (constraintsBuilder.isIndirectOutputOperand(i)) {
          indirectOutputLVals.push_back(lval);
        } else {
          outputLVals.push_back(lval);
          outputTypes.push_back(lval->getType()->getPointerElementType());
        }
      } else {
        inputArgs.push_back(e);
      }
    }
  }

  const size_t N = outputTypes.size();
  LLType *returnType =
      N == 0 ? llvm::Type::getVoidTy(irs->context())
             : N == 1 ? outputTypes[0]
                      : LLStructType::get(irs->context(), outputTypes);

  LLValue *rval = DtoInlineAsmExpr(stmt->loc, insn, constraints,
                                   indirectOutputLVals, inputArgs, returnType);

  if (N == 1) {
    DtoStore(rval, outputLVals[0]);
  } else {
    for (size_t i = 0; i < N; ++i) {
      auto element = DtoExtractValue(rval, i);
      DtoStore(element, outputLVals[i]);
    }
  }
}
