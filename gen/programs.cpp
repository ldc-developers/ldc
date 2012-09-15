#include "gen/programs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"

#include "root.h"       // error(char*)
#include "mars.h"       // fatal()

using namespace llvm;

static cl::opt<std::string> gcc("gcc",
    cl::desc("GCC to use for assembling and linking"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> ar("ar",
    cl::desc("Archiver"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> link("ms-link",
    cl::desc("LINK to use for linking on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> lib("ms-lib",
    cl::desc("Library Manager to use on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

sys::Path getProgram(const char *name, const cl::opt<std::string> &opt, const char *envVar = 0)
{
    const char *prog = NULL;

    if (opt.getNumOccurrences() > 0 && opt.length() > 0)
        prog = gcc.c_str();

    if (!prog && envVar)
        prog = getenv(envVar);
    if (!prog)
        prog = name;

    sys::Path path = sys::Program::FindProgramByName(prog);
    if (path.empty() && !prog) {
        if (prog) {
            path.set(prog);
        } else {
            error("failed to locate %s", name);
            fatal();
        }
    }

    return path;
}

sys::Path getGcc()
{
    return getProgram("gcc", gcc, "CC");
}

sys::Path getArchiver()
{
    return getProgram("ar", ar);
}

sys::Path getLink()
{
    return getProgram("link.exe", link);
}

sys::Path getLib()
{
    return getProgram("lib.exe", lib);
}
