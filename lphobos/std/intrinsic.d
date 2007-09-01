

// written by Walter Bright
// www.digitalmars.com
// Placed into the public domain

/* NOTE: This file has been patched from the original DMD distribution to
   work with the GDC compiler.
   NOTE: This file has been patched from the original GDC distribution to
   work with the LLVMDC compiler.

   Modified by David Friedman, May 2006
   Modified by Tomas Lindquist Olsen, August 2007
*/

/** These functions are built-in intrinsics to the compiler.
 *
	Intrinsic functions are functions built in to the compiler,
	usually to take advantage of specific CPU features that
	are inefficient to handle via external functions.
	The compiler's optimizer and code generator are fully
	integrated in with intrinsic functions, bringing to bear
	their full power on them.
	This can result in some surprising speedups.
 * Macros:
 *	WIKI=Phobos/StdIntrinsic
 */

module std.intrinsic;

/**
 * Scans the bits in v starting with bit 0, looking
 * for the first set bit.
 * Returns:
 *	The bit number of the first bit set.
 *	The return value is undefined if v is zero.
 */
version (LLVM)
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
else
    int bsf(uint v);

/**
 * Scans the bits in v from the most significant bit
 * to the least significant bit, looking
 * for the first set bit.
 * Returns:
 *	The bit number of the first bit set.
 *	The return value is undefined if v is zero.
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
version (LLVM)
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
else
    int bsr(uint v);

/**
 * Tests the bit.
 */
version (LLVM)
int bt(uint *p, uint bitnum)
{
    return (p[bitnum / (uint.sizeof*8)] & (1<<(bitnum & ((uint.sizeof*8)-1)))) ? -1 : 0 ;
}
else
    int bt(uint *p, uint bitnum);

/**
 * Tests and complements the bit.
 */
version (LLVM)
int btc(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q ^= mask;
    return result ? -1 : 0;
}
else
int btc(uint *p, uint bitnum);

/**
 * Tests and resets (sets to 0) the bit.
 */
version (LLVM)
int btr(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q &= ~mask;
    return result ? -1 : 0;
}
else
    int btr(uint *p, uint bitnum);

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
 * 	A non-zero value if the bit was set, and a zero
 *	if it was clear.
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
version (LLVM)
int bts(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q |= mask;
    return result ? -1 : 0;
}
else
    int bts(uint *p, uint bitnum);


/**
 * Swaps bytes in a 4 byte uint end-to-end, i.e. byte 0 becomes
	byte 3, byte 1 becomes byte 2, byte 2 becomes byte 1, byte 3
	becomes byte 0.
 */
version (LLVM)
pragma(LLVM_internal, "intrinsic", "llvm.bswap.i32.i32")
    uint bswap(uint val);
else
    uint bswap(uint v);


/**
 * Reads I/O port at port_address.
 */
version (LLVM)
    ubyte  inp(uint p) { return 0; }
else
    ubyte  inp(uint port_address);

/**
 * ditto
 */
version (LLVM)
    ushort inpw(uint p) { return 0; }
else
    ushort inpw(uint port_address);

/**
 * ditto
 */
version (LLVM)
    uint   inpl(uint p) { return 0; }
else
    uint   inpl(uint port_address);


/**
 * Writes and returns value to I/O port at port_address.
 */
version (LLVM)
    ubyte  outp(uint p, ubyte v) { return v; }
else
    ubyte  outp(uint port_address, ubyte value);

/**
 * ditto
 */
version (LLVM)
    ushort outpw(uint p, ushort v) { return v; }
else
    ushort outpw(uint port_address, ushort value);

/**
 * ditto
 */
version (LLVM)
    uint   outpl(uint p, uint v) { return v; }
else
    uint   outpl(uint port_address, uint value);


