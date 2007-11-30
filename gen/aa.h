#ifndef LLVMDC_GEN_AA_H
#define LLVMDC_GEN_AA_H

DValue* DtoAAIndex(Type* type, DValue* aa, DValue* key);
DValue* DtoAAIn(Type* type, DValue* aa, DValue* key);
void DtoAARemove(DValue* aa, DValue* key);

#endif // LLVMDC_GEN_AA_H
