module asm_diagnostics2;

void barTemplate()()
{
    import ldc.llvmasm;
    __asm(
        `hello
        movq 123, %elx`, "~{eax}");
}
