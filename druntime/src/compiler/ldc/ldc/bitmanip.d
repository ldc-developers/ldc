/*
 * D phobos intrinsics for LDC
 *
 * From GDC ... public domain!
 */
module ldc.bitmanip;

// Check for the right compiler
version(LDC)
{
    // OK
}
else
{
    static assert(false, "This module is only valid for LDC");
}

int bsf(uint v)
{
    uint m = 1;
    uint i;
    for (i = 0; i < 32; i++,m<<=1) {
        if (v&m)
        return i;
    }
    return i; // supposed to be undefined
}

int bsr(uint v)
{
    uint m = 0x80000000;
    uint i;
    for (i = 32; i ; i--,m>>>=1) {
    if (v&m)
        return i-1;
    }
    return i; // supposed to be undefined
}

int bt(uint *p, uint bitnum)
{
    return (p[bitnum / (uint.sizeof*8)] & (1<<(bitnum & ((uint.sizeof*8)-1)))) ? -1 : 0 ;
}

int btc(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q ^= mask;
    return result ? -1 : 0;
}

int btr(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q &= ~mask;
    return result ? -1 : 0;
}

int bts(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q |= mask;
    return result ? -1 : 0;
}

pragma(intrinsic, "llvm.bswap.i32")
    uint bswap(uint val);

ubyte  inp(uint p) { throw new Exception("inp intrinsic not yet implemented"); }
ushort inpw(uint p) { throw new Exception("inpw intrinsic not yet implemented"); }
uint   inpl(uint p) { throw new Exception("inpl intrinsic not yet implemented"); }

ubyte  outp(uint p, ubyte v) { throw new Exception("outp intrinsic not yet implemented"); }
ushort outpw(uint p, ushort v) { throw new Exception("outpw intrinsic not yet implemented"); }
uint   outpl(uint p, uint v) { throw new Exception("outpl intrinsic not yet implemented"); }
