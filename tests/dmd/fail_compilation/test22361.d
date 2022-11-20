/*
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
---
fail_compilation/test22361.d(9): Error: unable to read module `this_module_does_not_exist`
fail_compilation/test22361.d(9):        Expected 'this_module_does_not_exist.d' or 'this_module_does_not_exist/package.d' in one of the following import paths:
---
*/
import this_module_does_not_exist;
