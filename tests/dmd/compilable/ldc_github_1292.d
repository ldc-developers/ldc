// DISABLED: LDC_not_x86

void main()
{
    fun();
}

void fun()
{
    double x0 = 0,
           x1 = 1;

    asm nothrow @nogc
    {
        movlpd qword ptr x0, XMM0;
        movhpd qword ptr x1, XMM0;
    }
}