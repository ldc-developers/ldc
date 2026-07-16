// EXTRA_SOURCES: imports/你好.d
// UNICODE_NAMES:

// LDC: MS linker apparently doesn't (properly?) support Unicode in
//      `/INCLUDE:symbol` linker directives embedded in object files
//     (for llvm.used symbols). LLD works (-link-internally).
// REQUIRED_ARGS(windows): -link-internally

import imports.你好;

int main()
{
	return foo();
}
