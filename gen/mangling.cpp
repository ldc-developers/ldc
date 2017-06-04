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
#include "gen/to_string.h"
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
      for (auto package : *packages) {
        llvm::StringRef str = package->toChars();
        ret += ldc::to_string(str.size());
        ret += str;
      }
    }
    llvm::StringRef str = moddecl->id->toChars();
    ret += ldc::to_string(str.size());
    ret += str;
  }

  // source line number
  auto lineNo = ldc::to_string(symb->loc.linnum);
  ret += ldc::to_string(lineNo.size()+1);
  ret += 'L';
  ret += lineNo;

  // MD5 hash
  auto hashedName = hashName(name);
  ret += "33_"; // add underscore to delimit the 33 character count
  ret += hashedName;

  // top aggregate
  if (auto agg = symb->isMember()) {
    llvm::StringRef topaggr = agg->ident->toChars();
    ret += ldc::to_string(topaggr.size());
    ret += topaggr;
  }

  // identifier
  llvm::StringRef identifier = symb->toChars();
  ret += ldc::to_string(identifier.size());
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

  // TODO: Cache the result?

  return gABI->mangleFunctionForLLVM(std::move(mangledName), link);
}

std::string getMangledName(VarDeclaration *vd) {
  // TODO: is hashing of variable names necessary?

  // TODO: Cache the result?

  OutBuffer mangleBuf;
  mangleToBuffer(vd, &mangleBuf);

  return gABI->mangleVariableForLLVM(mangleBuf.peekString(), vd->linkage);
}

namespace {
std::string getMangledAggregateName(AggregateDeclaration *ad,
                                    const char *suffix) {
  std::string ret = "_D";

  OutBuffer mangleBuf;
  mangleToBuffer(ad, &mangleBuf);
  llvm::StringRef mangledAggrName = mangleBuf.peekString();

  if (shouldHashAggrName(mangledAggrName)) {
    ret += hashSymbolName(mangledAggrName, ad);
  } else {
    ret += mangledAggrName;
  }

  if (suffix)
    ret += suffix;

  return gABI->mangleVariableForLLVM(std::move(ret), LINKd);
}
}

std::string getMangledInitSymbolName(AggregateDeclaration *aggrdecl) {
  return getMangledAggregateName(aggrdecl, "6__initZ");
}

std::string getMangledVTableSymbolName(AggregateDeclaration *aggrdecl) {
  return getMangledAggregateName(aggrdecl, "6__vtblZ");
}

std::string getMangledClassInfoSymbolName(AggregateDeclaration *aggrdecl) {
  const char *suffix =
      aggrdecl->isInterfaceDeclaration() ? "11__InterfaceZ" : "7__ClassZ";
  return getMangledAggregateName(aggrdecl, suffix);
}
