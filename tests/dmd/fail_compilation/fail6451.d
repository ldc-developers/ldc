/*
DISABLED: win32 win64 linux32 osx32 freebsd32 LDC
TEST_OUTPUT:
---
fail_compilation/fail6451.d(9): Error: `__va_list_tag` is not defined, perhaps `import core.stdc.stdarg;` ?
---
*/

void error(...){}
