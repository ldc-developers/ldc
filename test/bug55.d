module bug55;
extern(C) int printf(char*, ...);

int atoi(char[] s) {
    int i, fac=1;
    bool neg = (s.length) && (s[0] == '-');
    char[] a = neg ? s[1..$] : s;
    foreach_reverse(c; a) {
        i += (c-'0') * fac;
        fac *= 10;
    }
    return !neg ? i : -i;
}

void main()
{
    printf("64213 = %d\n", atoi("64213"));
    printf("-64213 = %d\n", atoi("-64213"));
    assert(atoi("64213") == 64213);
    assert(atoi("-64213") == -64213);
}
