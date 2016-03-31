//===-- mangling.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Tries to centralize functionality for mangling of symbols.
//
//===----------------------------------------------------------------------===//

#include "gen/mangling.h"

#include "ddmd/declaration.h"
#include "ddmd/dsymbol.h"
#include "ddmd/identifier.h"
#include "ddmd/module.h"
#include "gen/abi.h"
#include "gen/irstate.h"
#include "llvm/Support/MD5.h"

namespace {

// TODO: Disable hashing of symbols that are defined in libdruntime and
// libphobos. This would enable hashing thresholds below the largest symbol in
// libdruntime/phobos.

bool shouldHashAggrName(llvm::StringRef name) {
  /// Add extra chars to the length of aggregate names to account for
  /// the additional D mangling suffix and prefix
  return (global.params.hashThreshold != 0) &&
         ((name.size() + 11) > global.params.hashThreshold);
}

llvm::SmallString<32> hashName(llvm::StringRef name) {
  llvm::MD5 hasher;
  hasher.update(name);
  llvm::MD5::MD5Result result;
  hasher.final(result);
  llvm::SmallString<32> hashStr;
  llvm::MD5::stringifyResult(result, hashStr);

  return hashStr;
}

/// Hashes the symbol name and prefixes the hash with some recognizable parts of
/// the full symbol name. The prefixing means that the hashed name may be larger
/// than the input when identifiers are very long and the hash threshold is low.
/// Demangled hashed name is:
/// module.L<line_no>.<hash>.<top aggregate>.<identifier>
std::string hashSymbolName(llvm::StringRef name, Dsymbol *symb) {
  std::string ret;

  // module
  {
    auto moddecl = symb->getModule()->md;
    assert(moddecl);
    if (auto packages = moddecl->packages) {
      for (size_t i = 0; i < packages->dim; ++i) {
        llvm::StringRef str = (*packages)[i]->toChars();
        ret += std::to_string(str.size());
        ret += str;
      }
    }
    llvm::StringRef str = moddecl->id->toChars();
    ret += std::to_string(str.size());
    ret += str;
  }

  // source line number
  auto lineNo = std::to_string(symb->loc.linnum);
  ret += std::to_string(lineNo.size()+1);
  ret += 'L';
  ret += lineNo;

  // MD5 hash
  auto hashedName = hashName(name);
  ret += "33_"; // add underscore to delimit the 33 character count
  ret += hashedName;

  // top aggregate
  if (auto agg = symb->isAggregateMember()) {
    llvm::StringRef topaggr = agg->ident->toChars();
    ret += std::to_string(topaggr.size());
    ret += topaggr;
  }

  // identifier
  llvm::StringRef identifier = symb->toChars();
  ret += std::to_string(identifier.size());
  ret += identifier;

  return ret;
}
}

std::string getMangledName(FuncDeclaration *fdecl, LINK link) {
  std::string mangledName(mangleExact(fdecl));

  // Hash the name if necessary
  if (((link == LINKd) || (link == LINKdefault)) &&
      (global.params.hashThreshold != 0) &&
      (mangledName.length() > global.params.hashThreshold)) {

    auto hashedName = hashSymbolName(mangledName, fdecl);
    mangledName = "_D" + hashedName + "Z";
  }

  return gABI->mangleForLLVM(mangledName, link);
}

std::string getMangledInitSymbolName(AggregateDeclaration *aggrdecl) {
  std::string ret = "_D";

  std::string mangledName = mangle(aggrdecl);
  if (shouldHashAggrName(mangledName)) {
    ret += hashSymbolName(mangledName, aggrdecl);
  } else {
    ret += mangledName;
  }

  ret += "6__initZ";

  return ret;
}

std::string getMangledVTableSymbolName(AggregateDeclaration *aggrdecl) {
  std::string ret = "_D";

  std::string mangledName = mangle(aggrdecl);
  if (shouldHashAggrName(mangledName)) {
    ret += hashSymbolName(mangledName, aggrdecl);
  } else {
    ret += mangledName;
  }

  ret += "6__vtblZ";

  return ret;
}

std::string getMangledClassInfoSymbolName(AggregateDeclaration *aggrdecl) {
  std::string ret = "_D";

  std::string mangledName = mangle(aggrdecl);
  if (shouldHashAggrName(mangledName)) {
    ret += hashSymbolName(mangledName, aggrdecl);
  } else {
    ret += mangledName;
  }

  if (aggrdecl->isInterfaceDeclaration()) {
    ret += "11__InterfaceZ";
  } else {
    ret += "7__ClassZ";
  }

  return ret;
}
