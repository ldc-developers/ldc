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

static cl::opt<std::string> mslink("ms-link",
    cl::desc("LINK to use for linking on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<std::string> mslib("ms-lib",
    cl::desc("Library Manager to use on Windows"),
    cl::Hidden,
    cl::ZeroOrMore);

sys::Path getProgram(const char *name, const cl::opt<std::string> &opt, const char *envVar = 0)
{
    sys::Path path;
    const char *prog = NULL;

    if (opt.getNumOccurrences() > 0 && opt.length() > 0 && (prog = opt.c_str()))
        path = sys::Program::FindProgramByName(prog);

    if (path.empty() && envVar && (prog = getenv(envVar)))
        path = sys::Program::FindProgramByName(prog);

    if (path.empty())
        path = sys::Program::FindProgramByName(name);

    if (path.empty()) {
        error("failed to locate %s", name);
        fatal();
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
    return getProgram("link.exe", mslink);
}

sys::Path getLib()
{
    return getProgram("lib.exe", mslib);
}
