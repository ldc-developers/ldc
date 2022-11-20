/*
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
---
fail_compilation/diag10327.d(10): Error: unable to read module `test10327`
fail_compilation/diag10327.d(10):        Expected 'imports/test10327.d' or 'imports/test10327/package.d' in one of the following import paths:
---
*/

import imports.test10327;  // package.d missing
