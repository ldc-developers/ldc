module mini.assign1;

extern(C) int printf(char*, ...);

struct X
{
    int a;
    alias a b;
}
void main()
{
    X e = void;
    e.a = e.b = 5;
    printf("%d - %d\n", e.a, e.b);
    assert(e.a == 5);
    assert(e.a == e.b);
}
