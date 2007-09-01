
/**
 *
	D constrains integral types to specific sizes. But efficiency
	of different sizes varies from machine to machine,
	pointer sizes vary, and the maximum integer size varies.
	<b>stdint</b> offers a portable way of trading off size
	vs efficiency, in a manner compatible with the <tt>stdint.h</tt>
	definitions in C.

	The exact aliases are types of exactly the specified number of bits.
	The at least aliases are at least the specified number of bits
	large, and can be larger.
	The fast aliases are the fastest integral type supported by the
	processor that is at least as wide as the specified number of bits.

	The aliases are:

	<table border=1 cellspacing=0 cellpadding=5>
	<th>Exact Alias
	<th>Description
	<th>At Least Alias
	<th>Description
	<th>Fast Alias
	<th>Description
	<tr>
	<td>int8_t
	<td>exactly 8 bits signed
	<td>int_least8_t
	<td>at least 8 bits signed
	<td>int_fast8_t
	<td>fast 8 bits signed
	<tr>
	<td>uint8_t
	<td>exactly 8 bits unsigned
	<td>uint_least8_t
	<td>at least 8 bits unsigned
	<td>uint_fast8_t
	<td>fast 8 bits unsigned

	<tr>
	<td>int16_t
	<td>exactly 16 bits signed
	<td>int_least16_t
	<td>at least 16 bits signed
	<td>int_fast16_t
	<td>fast 16 bits signed
	<tr>
	<td>uint16_t
	<td>exactly 16 bits unsigned
	<td>uint_least16_t
	<td>at least 16 bits unsigned
	<td>uint_fast16_t
	<td>fast 16 bits unsigned

	<tr>
	<td>int32_t
	<td>exactly 32 bits signed
	<td>int_least32_t
	<td>at least 32 bits signed
	<td>int_fast32_t
	<td>fast 32 bits signed
	<tr>
	<td>uint32_t
	<td>exactly 32 bits unsigned
	<td>uint_least32_t
	<td>at least 32 bits unsigned
	<td>uint_fast32_t
	<td>fast 32 bits unsigned

	<tr>
	<td>int64_t
	<td>exactly 64 bits signed
	<td>int_least64_t
	<td>at least 64 bits signed
	<td>int_fast64_t
	<td>fast 64 bits signed
	<tr>
	<td>uint64_t
	<td>exactly 64 bits unsigned
	<td>uint_least64_t
	<td>at least 64 bits unsigned
	<td>uint_fast64_t
	<td>fast 64 bits unsigned
	</table>

	The ptr aliases are integral types guaranteed to be large enough
	to hold a pointer without losing bits:

	<table border=1 cellspacing=0 cellpadding=5>
	<th>Alias
	<th>Description
	<tr>
	<td>intptr_t
	<td>signed integral type large enough to hold a pointer
	<tr>
	<td>uintptr_t
	<td>unsigned integral type large enough to hold a pointer
	</table>

	The max aliases are the largest integral types:

	<table border=1 cellspacing=0 cellpadding=5>
	<th>Alias
	<th>Description
	<tr>
	<td>intmax_t
	<td>the largest signed integral type
	<tr>
	<td>uintmax_t
	<td>the largest unsigned integral type
	</table>

 * Authors: Walter Bright, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdStdint
 */

/*
    NOTE: This file has been patched from the original DMD distribution to
    work with the LLVMDC compiler.

    Modified by Tomas Lindquist Olsen, August 2007
*/

module std.stdint;

/* Exact sizes */

alias  byte   int8_t;
alias ubyte  uint8_t;
alias  short  int16_t;
alias ushort uint16_t;
alias  int    int32_t;
alias uint   uint32_t;
alias  long   int64_t;
alias ulong  uint64_t;

/* At least sizes */

alias  byte   int_least8_t;
alias ubyte  uint_least8_t;
alias  short  int_least16_t;
alias ushort uint_least16_t;
alias  int    int_least32_t;
alias uint   uint_least32_t;
alias  long   int_least64_t;
alias ulong  uint_least64_t;

/* Fastest minimum width sizes */

alias  byte  int_fast8_t;
alias ubyte uint_fast8_t;
alias  int   int_fast16_t;
alias uint  uint_fast16_t;
alias  int   int_fast32_t;
alias uint  uint_fast32_t;
alias  long  int_fast64_t;
alias ulong uint_fast64_t;

/* Integer pointer holders */

version(LLVM64) {
    alias long   intptr_t;
    alias ulong uintptr_t;
}
else {
    alias int   intptr_t;
    alias uint uintptr_t;
}

/* Greatest width integer types */

alias  long  intmax_t;
alias ulong uintmax_t;

