//===-- programs.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/programs.h"
#include "mars.h"       // fatal()
#include "root.h"       // error(char*)
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"

using namespace llvm;

static cl::opt<std::string> gcc("gcc",
    cl::desc("GCC to use for assembling and linking"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> ar("ar",
    cl::desc("Archiver"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> mslink("ms-link",
    cl::desc("LINK to use for linking on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> mslib("ms-lib",
    cl::desc("Library Manager to use on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

#if LDC_LLVM_VER < 304
namespace llvm {
namespace sys {
inline std::string FindProgramByName(const std::string& name)
{
    return llvm::sys::Program::FindProgramByName(name).str();
}
} // namespace sys
} // namespace llvm
#endif

static std::string getProgram(const char *name, const cl::opt<std::string> &opt, const char *envVar = 0)
{
    std::string path;
    const char *prog = NULL;

    if (opt.getNumOccurrences() > 0 && opt.length() > 0 && (prog = opt.c_str()))
        path = sys::FindProgramByName(prog);

    if (path.empty() && envVar && (prog = getenv(envVar)))
        path = sys::FindProgramByName(prog);

    if (path.empty())
        path = sys::FindProgramByName(name);

    if (path.empty()) {
        error("failed to locate %s", name);
        fatal();
    }

    return path;
}

std::string getGcc()
{
    return getProgram("gcc", gcc, "CC");
}

std::string getArchiver()
{
    return getProgram("ar", ar);
}

std::string getLink()
{
    return getProgram("link.exe", mslink);
}

std::string getLib()
{
    return getProgram("lib.exe", mslib);
}
