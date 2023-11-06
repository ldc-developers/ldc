/*
TEST_OUTPUT:
---
fail_compilation/ldc_github_1822.d(13): Error: call to unimplemented abstract function `void coreDump()`
fail_compilation/ldc_github_1822.d(13):        declared here: fail_compilation/ldc_github_1822.d(19)
---
*/

class Child : Parent
{
    override void coreDump()
    {
        super.coreDump();
    }
}

class Parent
{
    abstract void coreDump();
}
