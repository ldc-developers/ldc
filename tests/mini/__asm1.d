import ldc.llvmasm;
void main() {
    version(X86)
    {
        int i;
        __asm("movl $1, $0", "=*m,i", &i, 42);
        assert(i == 42);

        int j = __asm!(int)("movl $1, %eax", "={ax},i", 42);
        assert(j == 42);

        auto k = __asmtuple!(int,int)("mov $2, %eax ; mov $3, %edx", "={ax},={dx},i,i", 10, 20);
        assert(k.v[0] == 10);
        assert(k.v[1] == 20);
    }
    else version(PPC)
    {
        int j = 42;
        int i = __asm!(int)("li $1, $0", "=r,*m", &j);
        assert(i == 42);
    }
}
