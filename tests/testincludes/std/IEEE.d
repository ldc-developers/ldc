// Written in the D programming language
/*
 * Authors:
 *      Walter Bright, Don Clugston
 * Copyright:
 *      Copyright (c) 2001-2005 by Digital Mars,
 *      All Rights Reserved,
 *      www.digitalmars.com
 * License:
 *  This software is provided 'as-is', without any express or implied
 *  warranty. In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  <ul>
 *  <li> The origin of this software must not be misrepresented; you must not
 *       claim that you wrote the original software. If you use this software
 *       in a product, an acknowledgment in the product documentation would be
 *       appreciated but is not required.
 *  </li>
 *  <li> Altered source versions must be plainly marked as such, and must not
 *       be misrepresented as being the original software.
 *  </li>
 *  <li> This notice may not be removed or altered from any source
 *       distribution.
 *  </li>
 *  </ul>
 */
/* Cut down version for libtangobos-partial/dstress */

module tango.math.IEEE;


private:
/*
 * The following IEEE 'real' formats are currently supported:
 * 64 bit Big-endian  'double' (eg PowerPC)
 * 128 bit Big-endian 'quadruple' (eg SPARC)
 * 64 bit Little-endian 'double' (eg x86-SSE2)
 * 80 bit Little-endian, with implied bit 'real80' (eg x87, Itanium).
 * 128 bit Little-endian 'quadruple' (not implemented on any known processor!)
 *
 * Non-IEEE 128 bit Big-endian 'doubledouble' (eg PowerPC) has partial support
 */
version(LittleEndian) {
    static assert(real.mant_dig == 53 || real.mant_dig==64
               || real.mant_dig == 113,
      "Only 64-bit, 80-bit, and 128-bit reals"
      " are supported for LittleEndian CPUs");
} else {
    static assert(real.mant_dig == 53 || real.mant_dig==106
               || real.mant_dig == 113,
    "Only 64-bit and 128-bit reals are supported for BigEndian CPUs."
    " double-double reals have partial support");
}

// Constants used for extracting the components of the representation.
// They supplement the built-in floating point properties.
template floatTraits(T) {
 // EXPMASK is a ushort mask to select the exponent portion (without sign)
 // POW2MANTDIG = pow(2, real.mant_dig) is the value such that
 //  (smallest_denormal)*POW2MANTDIG == real.min
 // EXPPOS_SHORT is the index of the exponent when represented as a ushort array.
 // SIGNPOS_BYTE is the index of the sign when represented as a ubyte array.
 static if (T.mant_dig == 24) { // float
    const ushort EXPMASK = 0x7F80;
    const ushort EXPBIAS = 0x3F00;
    const uint EXPMASK_INT = 0x7F80_0000;
    const uint MANTISSAMASK_INT = 0x007F_FFFF;
    const real POW2MANTDIG = 0x1p+24;
    version(LittleEndian) {
      const EXPPOS_SHORT = 1;
    } else {
      const EXPPOS_SHORT = 0;
    }
 } else static if (T.mant_dig == 53) { // double, or real==double
    const ushort EXPMASK = 0x7FF0;
    const ushort EXPBIAS = 0x3FE0;
    const uint EXPMASK_INT = 0x7FF0_0000;
    const uint MANTISSAMASK_INT = 0x000F_FFFF; // for the MSB only
    const real POW2MANTDIG = 0x1p+53;
    version(LittleEndian) {
      const EXPPOS_SHORT = 3;
      const SIGNPOS_BYTE = 7;
    } else {
      const EXPPOS_SHORT = 0;
      const SIGNPOS_BYTE = 0;
    }
 } else static if (T.mant_dig == 64) { // real80
    const ushort EXPMASK = 0x7FFF;
    const ushort EXPBIAS = 0x3FFE;
    const real POW2MANTDIG = 0x1p+63;
    version(LittleEndian) {
      const EXPPOS_SHORT = 4;
      const SIGNPOS_BYTE = 9;
    } else {
      const EXPPOS_SHORT = 0;
      const SIGNPOS_BYTE = 0;
    }
 } else static if (real.mant_dig == 113){ // quadruple
    const ushort EXPMASK = 0x7FFF;
    const real POW2MANTDIG = 0x1p+113;
    version(LittleEndian) {
      const EXPPOS_SHORT = 7;
      const SIGNPOS_BYTE = 15;
    } else {
      const EXPPOS_SHORT = 0;
      const SIGNPOS_BYTE = 0;
    }
 } else static if (real.mant_dig == 106) { // doubledouble
    const ushort EXPMASK = 0x7FF0;
    const real POW2MANTDIG = 0x1p+53;  // doubledouble denormals are strange
    // and the exponent byte is not unique
    version(LittleEndian) {
      const EXPPOS_SHORT = 7; // [3] is also an exp short
      const SIGNPOS_BYTE = 15;
    } else {
      const EXPPOS_SHORT = 0; // [4] is also an exp short
      const SIGNPOS_BYTE = 0;
    }
 }
}


public:
/*********************************
 * Return 1 if sign bit of e is set, 0 if not.
 */

int signbit(real x)
{
    return ((cast(ubyte *)&x)[floatTraits!(real).SIGNPOS_BYTE] & 0x80) != 0;
}
