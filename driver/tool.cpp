//===-- tool.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/tool.h"
#include "mars.h"
#include "llvm/Support/Program.h"

int executeToolAndWait(llvm::sys::Path tool, std::vector<std::string> const & args, bool verbose)
{
    // Construct real argument list.
    // First entry is the tool itself, last entry must be NULL.
    std::vector<const char *> realargs;
    realargs.reserve(args.size() + 2);
    realargs.push_back(tool.c_str());
    for (std::vector<std::string>::const_iterator it = args.begin(); it != args.end(); ++it)
    {
        realargs.push_back((*it).c_str());
    }
    realargs.push_back(NULL);

    // Print command line if requested
    if (verbose)
    {
        // Print it
        for (size_t i = 0; i < realargs.size()-1; i++)
            printf("%s ", realargs[i]);
        printf("\n");
        fflush(stdout);
    }

    // Execute tool.
    std::string errstr;
    if (int status = llvm::sys::Program::ExecuteAndWait(tool, &realargs[0], NULL, NULL, 0, 0, &errstr))
    {
        error("%s failed with status: %d", tool.c_str(), status);
        if (!errstr.empty())
            error("message: %s", errstr.c_str());
        return status;
    }
    return 0;
}
