// D import file generated from 'bitmanip.d'
module ldc.bitmanip;
version (LDC)
{
}
else
{
    static assert(false,"This module is only valid for LDC");
}
int bsf(uint v);
int bsr(uint v);
int bt(uint* p, uint bitnum)
{
return p[bitnum / ((uint).sizeof * 8)] & 1 << (bitnum & (uint).sizeof * 8 - 1) ? -1 : 0;
}
int btc(uint* p, uint bitnum)
{
uint* q = p + bitnum / ((uint).sizeof * 8);
uint mask = 1 << (bitnum & (uint).sizeof * 8 - 1);
int result = *q & mask;
*q ^= mask;
return result ? -1 : 0;
}
int btr(uint* p, uint bitnum)
{
uint* q = p + bitnum / ((uint).sizeof * 8);
uint mask = 1 << (bitnum & (uint).sizeof * 8 - 1);
int result = *q & mask;
*q &= ~mask;
return result ? -1 : 0;
}
int bts(uint* p, uint bitnum)
{
uint* q = p + bitnum / ((uint).sizeof * 8);
uint mask = 1 << (bitnum & (uint).sizeof * 8 - 1);
int result = *q & mask;
*q |= mask;
return result ? -1 : 0;
}
pragma (intrinsic, "llvm.bswap.i32")
{
    uint bswap(uint val);
}
ubyte inp(uint p);
ushort inpw(uint p);
uint inpl(uint p);
ubyte outp(uint p, ubyte v);
ushort outpw(uint p, ushort v);
uint outpl(uint p, uint v);
