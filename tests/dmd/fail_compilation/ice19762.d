// This bug wasn't fixed properly, so I've reopened https://issues.dlang.org/show_bug.cgi?id=19762.
// DISABLED: LDC

// EXTRA_FILES: imports/b19762.d imports/c19762.d
// PERMUTE_ARGS: -g

/*
TEST_OUTPUT:
---
fail_compilation/ice19762.d(13): Error: struct `ice19762.X` had semantic errors when compiling
---
*/

module ice19762;

struct X
{
	import imports.b19762 : Baz;
	Err err;
}
