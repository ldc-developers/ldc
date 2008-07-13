extern(C) int printf(char*, ...);

int return_six()
{
    return 6;
}

int add_int(int a, int b)
{
    return a+b;
}

int sub_int(int a, int b)
{
    return a-b;
}

alias int function(int,int) binfn_t;

int binop_int(binfn_t op, int a, int b)
{
    return op(a,b);
}

binfn_t get_binop_int(char op)
{
    binfn_t fn;
    if (op == '+')
        fn = &add_int;
    else if (op == '-')
        fn = &sub_int;
    return fn;
}

extern(C) float mul_float(float a, float b)
{
    return a * b;
}

void function_pointers()
{
    int function() fn = &return_six;
    assert(fn() == 6);

    binfn_t binfn = &add_int;
    assert(binfn(4,1045) == 1049);

    assert(binop_int(binfn, 10,656) == 666);

    binfn = get_binop_int('+');
    assert(binop_int(binfn, 10,100) == 110);
    binfn = get_binop_int('-');
    assert(binop_int(binfn, 10,100) == -90);

    {
    auto ffn = &mul_float;
    float ftmp = mul_float(2.5,5);
    assert(ftmp == 12.5);
    assert(ftmp > 12.49 && ftmp < 12.51);
    }
}

void main()
{
    printf("Function pointer test\n");
    function_pointers();
    printf("  SUCCESS\n");
}
