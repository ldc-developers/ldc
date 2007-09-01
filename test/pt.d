int main()
{
    char[16] s = void;
    {
    char[] sd = s;
    {
    s[0] = 'a';
    s[1] = 'b';
    s[2] = 'c';
    }

    printf("%p %p\n", s.ptr, sd.ptr);
    printf("%c%c%c\n", s[0], s[1], s[2]);
    }
    

    char[16] s1 = void;
    char[16] s2 = void;
    char[] d1 = s1;

    {
        printf("%p\n%p\n%p\n", s1.ptr, s2.ptr, d1.ptr);
    }

    int[16] arr;

    return 0;
}
