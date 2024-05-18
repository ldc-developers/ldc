/* TEST_OUTPUT:
---
fail_compilation/fail17612.d(17): Error: undefined identifier `string`
fail_compilation/fail17612.d(20): Error: `TypeInfo` not found. object.d may be incorrectly installed or corrupt.
fail_compilation/fail17612.d(20):        ldc2 might not be correctly installed.
fail_compilation/fail17612.d(20):        Please check your ldc2.conf configuration file.
fail_compilation/fail17612.d(20):        Installation instructions can be found at http://wiki.dlang.org/LDC.
---
*/

// https://issues.dlang.org/show_bug.cgi?id=17612

module object;

class Object
{
    string toString();
}

class TypeInfo {}
