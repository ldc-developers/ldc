module arrays13;

extern(C) int printf(char*, ...);

void main()
{
    char[] a = "hello";

    assert(a >  "hel");
    assert(a >= "hel");
    assert(a <  "helloo");
    assert(a <= "helloo");
    assert(a >  "betty");
    assert(a >= "betty");
    assert(a == "hello");
    assert(a <= "hello");
    assert(a >= "hello");
}
