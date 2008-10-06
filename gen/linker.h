#ifndef LDC_GEN_LINKER_H
#define LDC_GEN_LINKER_H

#include <vector>

namespace llvm
{
    class Module;
}

/**
 * Links the modules given in MV in to dst.
 * @param dst Destination module.
 * @param MV Vector of modules to link in to destination.
 */
void linkModules(llvm::Module* dst, const std::vector<llvm::Module*>& MV);

/**
 * Link an executable.
 * @param argv0 the argv[0] value as passed to main
 * @return 0 on success.
 */
int linkExecutable(const char* argv0);

/**
 * Delete the executable that was previously linked with linkExecutable.
 */
void deleteExecutable();

/**
 * Runs the executable that was previously linked with linkExecutable.
 * @return the return status of the executable.
 */
int runExectuable();

#endif // LDC_GEN_LINKER_H
