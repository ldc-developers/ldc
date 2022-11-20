// https://issues.dlang.org/show_bug.cgi?id=3004
/*
REQUIRED_ARGS: -ignore -v
LDC:  additionally exclude 'GC stats' line
TRANSFORM_OUTPUT: remove_lines("^(predefs|binary|version|config|DFLAG|parse|import|semantic|entry|library|function  object|function  core|GC stats|\s*$)")
TEST_OUTPUT:
---
pragma    GNU_attribute (__error)
pragma    GNU_attribute (__error)
code      test3004
---
*/

extern(C) int printf(char*, ...);

pragma(GNU_attribute, flatten)
void test() { printf("Hello GNU world!\n".dup.ptr); }

pragma(GNU_attribute, flatten);
