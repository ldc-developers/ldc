extern(C) int printf(char*, ...);

int main(char[][] args)
{
    printf("getint\n");
    int i = getint();
    printf("assert true\n");
    assert(i == 1234);
    printf("assert false\n");
    assert(i != 1234);
    printf("return\n");
    return 0;
}

int getint()
{
    return 1234;
}
