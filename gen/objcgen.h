#ifndef LDC_GEN_OBJCGEN_H
#define LDC_GEN_OBJCGEN_H

#include "gen/llvm.h"

class Type;
struct ObjcSelector;

void objc_init();
const char* objc_getMsgSend(Type *ret, bool hasHiddenArg);
void objc_Module_genmoduleinfo_classes();
LLGlobalVariable *objc_getMethVarRef(const ObjcSelector &sel);

#endif // LDC_GEN_OBJCGEN_H
