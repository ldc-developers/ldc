module switch3;
extern(C) int printf(char*, ...);

void main()
{
    char[] str = "hello";
    int i;
    switch(str)
    {
    case "world":
        i = 1;
        assert(0);
    case "hello":
        i = 2;
        break;
    case "a","b","c":
        i = 3;
        assert(0);
    default:
        i = 4;
        assert(0);
    }
    assert(i == 2);
    printf("SUCCESS\n");
}
