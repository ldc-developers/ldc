module bug32;

struct S
{
    char[] getName() { return name; }
    char[] name;
}

void main()
{
    S s = S("Kyle");
    char[] name = s.name;
    printf("%.*s\n", name.length, name.ptr);
}
