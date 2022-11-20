/*
EXTRA_FILES: imports/ice7782algorithm.d imports/ice7782range.d
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
----
fail_compilation/ice7782.d(12): Error: unable to read module `ice7782math`
fail_compilation/ice7782.d(12):        Expected 'imports/ice7782range/imports/ice7782math.d' or 'imports/ice7782range/imports/ice7782math/package.d' in one of the following import paths:
----
*/

import imports.ice7782algorithm;
import imports.ice7782range. imports.ice7782math;

void main() {}
