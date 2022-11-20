/*
REQUIRED_ARGS: -Xifoo
LDC: just a different error msg
DISABLED: LDC
TEST_OUTPUT:
---
Error: unrecognized switch '-Xifoo'
       run `dmd` to print the compiler manual
       run `dmd -man` to open browser on manual
---
*/
