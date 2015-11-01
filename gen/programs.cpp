//===-- programs.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/programs.h"
#include "mars.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"

using namespace llvm;

static cl::opt<std::string>
    gcc("gcc", cl::desc("GCC to use for assembling and linking"), cl::Hidden,
        cl::ZeroOrMore);

static cl::opt<std::string> ar("ar", cl::desc("Archiver"), cl::Hidden,
                               cl::ZeroOrMore);

static std::string findProgramByName(const std::string &name) {
#if LDC_LLVM_VER >= 306
  llvm::ErrorOr<std::string> res = llvm::sys::findProgramByName(name);
  return res ? res.get() : std::string();
#else
  return llvm::sys::FindProgramByName(name);
#endif
}

static std::string getProgram(const char *name, const cl::opt<std::string> *opt,
                              const char *envVar = NULL) {
  std::string path;
  const char *prog = NULL;

  if (opt && opt->getNumOccurrences() > 0 && opt->length() > 0 &&
      (prog = opt->c_str()))
    path = findProgramByName(prog);

  if (path.empty() && envVar && (prog = getenv(envVar)) && prog[0] != '\0')
    path = findProgramByName(prog);

  if (path.empty())
    path = findProgramByName(name);

  if (path.empty()) {
    error(Loc(), "failed to locate %s", name);
    fatal();
  }

  return path;
}

std::string getProgram(const char *name, const char *envVar) {
  return getProgram(name, NULL, envVar);
}

std::string getGcc() {
#if defined(__FreeBSD__) && __FreeBSD__ >= 10
  // Default compiler on FreeBSD 10 is clang
  return getProgram("clang", &gcc, "CC");
#else
  return getProgram("gcc", &gcc, "CC");
#endif
}

std::string getArchiver() { return getProgram("ar", &ar); }
