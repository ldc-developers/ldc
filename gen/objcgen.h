#ifndef LDC_GEN_OBJCGEN_H
#define LDC_GEN_OBJCGEN_H

#include "gen/llvm.h"

class Type;

void objc_init();
const char* objc_getMsgSend(Type *ret, bool hasHiddenArg);
void objc_Module_genmoduleinfo_classes();
llvm::GlobalVariable *objc_getMethVarRef(const char *s, size_t len);

#endif // LDC_GEN_OBJCGEN_H
