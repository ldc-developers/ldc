#ifndef LDC_GEN_AA_H
#define LDC_GEN_AA_H

DValue* DtoAAIndex(Loc& loc, Type* type, DValue* aa, DValue* key, bool lvalue);
DValue* DtoAAIn(Loc& loc, Type* type, DValue* aa, DValue* key);
void DtoAARemove(Loc& loc, DValue* aa, DValue* key);

#endif // LDC_GEN_AA_H
