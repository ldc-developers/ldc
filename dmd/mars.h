
#if IN_LLVM
#include "root/filename.h" // for `Strings`

struct Param;

int mars_mainBody(Param &params, Strings &files, Strings &libmodules);

void parseTransitionOption(Param &params, const char *name);
void parsePreviewOption(Param &params, const char *name);
void parseRevertOption(Param &params, const char *name);
#endif
