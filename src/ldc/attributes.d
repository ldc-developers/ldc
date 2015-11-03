/**
 * Contains compiler-recognized user-defined attribute types.
 *
 * Copyright: David Nadlinger 2015-2015
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   David Nadlinger
 */
module ldc.attributes;

/**
 * When applied to a variable, causes it to be emitted to a non-standard object
 * file/executable section.
 *
 * The target platform might impose certain restrictions on the format for
 * section names.
 * 
 * Examples:
 * ---
 * import ldc.attributes;
 * 
 * @section(".mySection") int myGlobal;
 * ---
 */
struct section {
	string name;
}
