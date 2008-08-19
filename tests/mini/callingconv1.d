module mini.callingconv1;

extern(C) int printf(char*, ...);

float foo(float a, float b)
{
    return a + b;
}

void main()
{
    float a = 1.5;
    float b = 2.5;
    float c;

    asm
    {
        mov EAX, [a];
        push EAX;
        mov EAX, [b];
        push EAX;
        call foo;
        fstp c;
    }

    printf("%f\n", c);
    
    assert(c == 4.0);
    
    printf("passed\n", c);
}
