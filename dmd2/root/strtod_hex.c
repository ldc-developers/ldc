//===-- strtod_hex.c ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//
//
// MSVC's strtod does not support hexadecimal floating points before MSVC 2014.
// strtod_hex is a hexadecimal float literal parser to be used within port.c
//
//===----------------------------------------------------------------------===//

#if _MSC_VER <= 1800

#include <cctype>
#include "longdouble.h"

longdouble strtod_hex(const char* p, char** endp) {
// References: http://dlang.org/lex.html
//             http://www.exploringbinary.com/hexadecimal-floating-point-constants
//             pages 57-58 of the C99 specification (http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf)
//             http://www.cplusplus.com/reference/cstdlib/strtod

// perhaps this function can rely on correctly formed hex floats (dmd2 front-end lexer), but I am not sure

    while (isspace(*p)) p++;

    if ( (p[0] != '0') || ((p[1] != 'x') && (p[1] != 'X')) )
        return 0;

    p += 2;

    bool sign = false; // positive
    unsigned num_digits = 0;
    bool dot = false;
    int exponent = -4;
    unsigned __int64 fraction = 0;
    while (true) {
        char c = *p;
        if (c == '_') {
            ++p;
        }
        else if (isxdigit(c)) {
            unsigned val = isalpha(c) ? ((c|0x20) - ('a'-10)) : (c - '0');
            if (num_digits < 14) {
                fraction = (fraction << 4) + val;
                if (fraction) { // ignore leading zeros
                    ++num_digits;
                    if (!dot) {
                        exponent += 4;
                    }
                }
            } else {
                if (!dot) {
                    exponent += 4;
                }
            }
            ++p;
        }
        else if ( (c == '.') && !dot ) {
            dot = true;
            ++p;
        } 
        else {
            break;
        }
    }

    if (!num_digits) {
        // the input is "0x" without valid digits after
        return 0;
    }

    // check for exponent (decimal number)
    int explicit_exponent = 0;
    if ((*p == 'p') || (*p == 'P')) {
        ++p;

        bool expsign = false;
        if (p[0]=='+')
            ++p;
        else if (*p =='-') {
            expsign = true;
            ++p;
        }

        while (true) {
            char c = *p;
            if (c == '_') {
                ++p;
            }
            else if (isdigit(c)) { // the exponent is decimal, no hex allowed!
                unsigned val = c - '0';
                explicit_exponent = explicit_exponent*10 + val;
                ++p;
            } 
            else {
                break;
            }
        }

        if (expsign) explicit_exponent *= -1;
    } 
    else {
        // handle error here? I believe that: dlang spec mandates explicit exponent, but C99 doesn't
    }

    if ((*p == 'f') || (*p == 'F') || (*p == 'l') || (*p == 'L')) {
        // handle trailing L or F (F = rounding needed at bitlocation where 32-bit float would truncate?)
        ++p;
    }

    if (endp) {
        *endp = (char*) p; // ugly cast needed to remove 'const'
    }
    

    fraction = fraction << ((14 - num_digits)*4);

    unsigned __int64 excess = fraction >> 53;
    while (excess) {
        ++exponent;
        fraction = fraction >> 1;
        excess = excess >> 1;
    }

    // exponent has 1023 offset (zero == 1023)
    exponent = explicit_exponent + exponent + 1023;

/*
 IEEE-754 64-bit floating point (from MSB to LSB) =
     1 sign bit
    11 bit exponent
    52 bit fraction (52/4 = 13 hex digits max + leading 1)
*/


    unsigned __int64 u = (sign ? 0x8000000000000000 : 0)
                       | ( *(unsigned __int64*)(&exponent) & (0x7FF)) << 52
                       | fraction & (0x000FFFFFFFFFFFFF); 

    return *(double*)&u;
}
#endif