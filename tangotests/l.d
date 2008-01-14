import tango.io.Console;

void main()
{
    printf("enter\n");
    assert(Cout !is null);
    Cout("Hi, says LLVMDC + Tango").newline;
    printf("exit\n");
}

extern(C) int printf(char*,...);
