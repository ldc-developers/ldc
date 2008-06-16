module typeinfo;
extern(C) int printf(char*, ...);

void main()
{
    auto ti = typeid(int);
    char[] str = ti.toString();
    printf("%.*s\n", str.length, str.ptr);
    assert(str == "int");
}
