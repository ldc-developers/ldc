//===-- asm-gcc.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// Converts a GDC/GCC-style inline assembly statement to an LLVM inline
// assembler expression.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "dmd/expression.h"
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

// Translates a GCC inline asm template string to LLVM's expected format.
std::string translateTemplate(GccAsmStatement *stmt) {
  const auto insn = peekString(stmt->insn->isStringExp());
  const auto N = insn.size();

  std::string result;
  result.reserve(static_cast<size_t>(N * 1.2));

  for (size_t i = 0; i < N; ++i) {
    const char c = insn[i];
    switch (c) {
    case '$':
      result += "$$"; // escape for LLVM: $ => $$
      break;
    case '%':
      if (i < N - 1 && insn[i + 1] == '%') { // unescape for LLVM: %% => %
        result += '%';
        ++i;
      } else {
        result += '$'; // e.g., %0 => $0
      }
      break;
    default:
      result += c;
      break;
    }
  }

  return result;
}

class ConstraintsBuilder {
  bool isAnyX86;
  std::ostringstream str; // LLVM constraints string being built
  LLSmallVector<bool, 8> _isIndirectOperand;

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
      isIndirect = true; // delay the commit
      code = code.substr(1);
    }

    const std::string name = translateName(code, isIndirect);

    if (isIndirect)
      str << '*';
    str << name;
    str << ',';

    return isIndirect;
  }

  // Might set `isIndirect` to true (but never resets to false).
  std::string translateName(llvm::StringRef gccName, bool &isIndirect) const {
    // clang translates GCC `m` to LLVM `*m` (indirect operand)
    if (gccName == "m") {
      isIndirect = true;
      return "m";
    }

    // some variable-width x86[_64] GCC register names aren't supported by LLVM
    // directly
    if (isAnyX86 && gccName.size() == 1) {
      switch (gccName[0]) {
      case 'a':
        return "{ax}";
      case 'b':
        return "{bx}";
      case 'c':
        return "{cx}";
      case 'd':
        return "{dx}";
      case 'S':
        return "{si}";
      case 'D':
        return "{di}";
      default:
        break;
      }
    }

    return needsCurlyBraces(gccName) ? ("{" + gccName + "}").str()
                                     : gccName.str();
  }

  // Register names need to be enclosed in curly braces for LLVM.
  bool needsCurlyBraces(llvm::StringRef gccName) const {
    auto N = gccName.size();
    if (N == 1 || (N == 3 && gccName[0] == '^'))
      return false;
    return !gccName.contains('{');
  }

public:
  ConstraintsBuilder() {
    auto arch = global.params.targetTriple->getArch();
    isAnyX86 = (arch == llvm::Triple::x86 || arch == llvm::Triple::x86_64);
  }

  // Returns the final constraints string for LLVM for a GCC-style asm
  // statement.
  std::string build(GccAsmStatement *stmt) {
    str.clear();
    _isIndirectOperand.clear();

    if (auto c = stmt->constraints) {
      _isIndirectOperand.reserve(c->length);
      for (size_t i = 0; i < c->length; ++i) {
        bool isOutput = (i < stmt->outputargs);
        bool isIndirect = append((*c)[i], isOutput ? '=' : 0);
        _isIndirectOperand.push_back(isIndirect);
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

  bool isIndirectOperand(size_t operandIndex) const {
    return _isIndirectOperand[operandIndex];
  }
};
}

void GccAsmStatement_toIR(GccAsmStatement *stmt, IRState *irs) {
  IF_LOG Logger::println("GccAsmStatement::toIR(): %s", stmt->loc.toChars());
  LOG_SCOPE;

  if (stmt->labels) {
    error(stmt->loc,
          "goto labels for GCC-style asm statements are not supported yet");
    fatal();
  }
  if (stmt->names) {
    for (Identifier *name : *stmt->names) {
      if (name) {
        error(stmt->loc,
              "symbolic names for operands in GCC-style assembly are not "
              "supported yet");
        fatal();
      }
    }
  }

  const std::string insn = translateTemplate(stmt);

  ConstraintsBuilder constraintsBuilder;
  const std::string constraints = constraintsBuilder.build(stmt);

  LLSmallVector<LLValue *, 8> outputLVals;
  LLSmallVector<LLType *, 8> outputTypes;
  LLSmallVector<LLType *, 8> indirectTypes;
  LLSmallVector<LLValue *, 8> operands;
  if (stmt->args) {
    for (size_t i = 0; i < stmt->args->length; ++i) {
      Expression *e = (*stmt->args)[i];
      const bool isOutput = (i < stmt->outputargs);
      const bool isIndirect = constraintsBuilder.isIndirectOperand(i);

      if (isOutput) {
        assert(e->isLvalue() && "should have been caught by front-end");
        LLValue *lval = DtoLVal(e);
        if (isIndirect) {
          operands.push_back(lval);
          indirectTypes.push_back(DtoType(e->type));
        } else {
          outputLVals.push_back(lval);
          outputTypes.push_back(DtoType(e->type));
        }
      } else {
        if (isIndirect && !e->isLvalue()) {
          error(e->loc,
                "indirect `\"m\"` input operands require an lvalue, but `%s` "
                "is an rvalue",
                e->toChars());
          fatal();
        }

        LLValue *inputVal = isIndirect ? DtoLVal(e) : DtoRVal(e);
        operands.push_back(inputVal);
        if (isIndirect)
          indirectTypes.push_back(DtoType(e->type));
      }
    }
  }

  const size_t N = outputTypes.size();
  LLType *returnType =
      N == 0 ? llvm::Type::getVoidTy(irs->context())
             : N == 1 ? outputTypes[0]
                      : LLStructType::get(irs->context(), outputTypes);

  LLValue *rval =
      DtoInlineAsmExpr(stmt->loc, insn, constraints, operands,
                       indirectTypes, returnType);

  if (N == 1) {
    DtoStore(rval, outputLVals[0]);
  } else {
    for (size_t i = 0; i < N; ++i) {
      auto element = DtoExtractValue(rval, i);
      DtoStore(element, outputLVals[i]);
    }
  }
}
