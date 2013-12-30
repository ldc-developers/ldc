module ldc.attribute;

/**
 * Compiler attributes.
 *
 * Copyright: Copyright The LDC Developers 2013
 * License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
 * Authors:   Kai Nacke <kai@redstar.de>
 */

/*          Copyright The LDC Developers 2013.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */


// Container for the passed attribute and arguments
private struct Attribute(A...)
{
    A args;
}

// The first argument names the attribute. The other arguments are parameters.
// Examples:
// @attribute("noinline") : function is never inlined
// @attribute("alignstack", 64) : set stack pointer alignment for function
// @attribute("section", ".interrupt_vector") : put function/variable into section .interrupt_vector
auto attribute(A...)(A args) if(A.length > 0 && is(A[0] == string))
{
    return Attribute!A(args);
}
