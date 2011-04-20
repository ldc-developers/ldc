#include "gen/programs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/System/Program.h"

#include "root.h"       // error(char*)
#include "mars.h"       // fatal()

using namespace llvm;

static cl::opt<std::string> gcc("gcc",
    cl::desc("GCC to use for assembling and linking"),
    cl::Hidden,
    cl::ZeroOrMore);


sys::Path getGcc() {
    const char *cc = NULL;
    
    if (gcc.getNumOccurrences() > 0 && gcc.length() > 0)
        cc = gcc.c_str();
    
    if (!cc)
        cc = getenv("CC");
    if (!cc)
        cc = "gcc";
    
    sys::Path path = sys::Program::FindProgramByName(cc);
    if (path.empty() && !cc) {
        if (cc) {
            path.set(cc);
        } else {
            error("failed to locate gcc");
            fatal();
        }
    }
    
    return path;
}
