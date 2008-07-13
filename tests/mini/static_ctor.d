extern(C) int printf(char*, ...);

static this()
{
    printf("static this\n");
}

static ~this()
{
    printf("static ~this\n");
}

void main()
{
    printf("main\n");
}
