/*
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
---
fail_compilation/fail21045.d(10): Error: unable to read module `__stdin`
fail_compilation/fail21045.d(10):        Expected '__stdin.d' or '__stdin/package.d' in one of the following import paths:
---
*/

import __stdin;
