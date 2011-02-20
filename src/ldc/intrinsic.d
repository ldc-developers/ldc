/*
 * D phobos intrinsics for LDC
 *
 * From GDC ... public domain!
 */
module std.intrinsic;

// Check for the right compiler
version(LDC)
{
    // OK
}
else
{
    static assert(false, "This module is only valid for LDC");
}

nothrow:

/**
 * Scans the bits in v starting with bit 0, looking
 * for the first set bit.
 * Returns:
 *      The bit number of the first bit set.
 *      The return value is undefined if v is zero.
 */
pure int bsf(size_t v)
{
    uint m = 1;
    uint i;
    for (i = 0; i < 32; i++,m<<=1) {
        if (v&m)
        return i;
    }
    return i; // supposed to be undefined
}

/**
 * Scans the bits in v from the most significant bit
 * to the least significant bit, looking
 * for the first set bit.
 * Returns:
 *      The bit number of the first bit set.
 *      The return value is undefined if v is zero.
 * Example:
 * ---
 * import std.intrinsic;
 *
 * int main()
 * {
 *     uint v;
 *     int x;
 *
 *     v = 0x21;
 *     x = bsf(v);
 *     printf("bsf(x%x) = %d\n", v, x);
 *     x = bsr(v);
 *     printf("bsr(x%x) = %d\n", v, x);
 *     return 0;
 * }
 * ---
 * Output:
 *  bsf(x21) = 0<br>
 *  bsr(x21) = 5
 */
pure int bsr(size_t v)
{
    uint m = 0x80000000;
    uint i;
    for (i = 32; i ; i--,m>>>=1) {
    if (v&m)
        return i-1;
    }
    return i; // supposed to be undefined
}


/**
 * Tests the bit.
 */
pure int bt(in size_t* p, size_t bitnum)
{
    return (p[bitnum / (uint.sizeof*8)] & (1<<(bitnum & ((uint.sizeof*8)-1)))) ? -1 : 0 ;
}


/**
 * Tests and complements the bit.
 */
int btc(size_t* p, size_t bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q ^= mask;
    return result ? -1 : 0;
}


/**
 * Tests and resets (sets to 0) the bit.
 */
int btr(size_t* p, size_t bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q &= ~mask;
    return result ? -1 : 0;
}


/**
 * Tests and sets the bit.
 * Params:
 * p = a non-NULL pointer to an array of uints.
 * index = a bit number, starting with bit 0 of p[0],
 * and progressing. It addresses bits like the expression:
---
p[index / (uint.sizeof*8)] & (1 << (index & ((uint.sizeof*8) - 1)))
---
 * Returns:
 *      A non-zero value if the bit was set, and a zero
 *      if it was clear.
 *
 * Example:
 * ---
import std.intrinsic;

int main()
{
    uint array[2];

    array[0] = 2;
    array[1] = 0x100;

    printf("btc(array, 35) = %d\n", <b>btc</b>(array, 35));
    printf("array = [0]:x%x, [1]:x%x\n", array[0], array[1]);

    printf("btc(array, 35) = %d\n", <b>btc</b>(array, 35));
    printf("array = [0]:x%x, [1]:x%x\n", array[0], array[1]);

    printf("bts(array, 35) = %d\n", <b>bts</b>(array, 35));
    printf("array = [0]:x%x, [1]:x%x\n", array[0], array[1]);

    printf("btr(array, 35) = %d\n", <b>btr</b>(array, 35));
    printf("array = [0]:x%x, [1]:x%x\n", array[0], array[1]);

    printf("bt(array, 1) = %d\n", <b>bt</b>(array, 1));
    printf("array = [0]:x%x, [1]:x%x\n", array[0], array[1]);

    return 0;
}
 * ---
 * Output:
<pre>
btc(array, 35) = 0
array = [0]:x2, [1]:x108
btc(array, 35) = -1
array = [0]:x2, [1]:x100
bts(array, 35) = 0
array = [0]:x2, [1]:x108
btr(array, 35) = -1
array = [0]:x2, [1]:x100
bt(array, 1) = -1
array = [0]:x2, [1]:x100
</pre>
 */
int bts(size_t* p, size_t bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q |= mask;
    return result ? -1 : 0;
}

/**
 * Swaps bytes in a 4 byte uint end-to-end, i.e. byte 0 becomes
 * byte 3, byte 1 becomes byte 2, byte 2 becomes byte 1, byte 3
 * becomes byte 0.
 */
pure pragma(intrinsic, "llvm.bswap.i32")
    uint bswap(uint val);

/**
 * Reads I/O port at port_address.
 */
ubyte inp(uint port_address) { assert(false && "inp intrinsic not yet implemented"); };

/**
 * ditto
 */
ushort inpw(uint port_address) { assert(false && "inpw intrinsic not yet implemented"); };

/**
 * ditto
 */
uint inpl(uint port_address) { assert(false && "inpl intrinsic not yet implemented"); };


/**
 * Writes and returns value to I/O port at port_address.
 */
ubyte outp(uint port_address, ubyte value) { assert(false && "outp intrinsic not yet implemented"); };

/**
 * ditto
 */
ushort outpw(uint port_address, ushort value) { assert(false && "outpw intrinsic not yet implemented"); };

/**
 * ditto
 */
uint outpl(uint port_address, uint value) { assert(false && "outpl intrinsic not yet implemented"); };
