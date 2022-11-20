// DISABLED: LDC // broken for dmd, too: https://issues.dlang.org/show_bug.cgi?id=15943
__gshared private:
    int j;
    extern(C++, ns) int k;

void f()
{
    j = 0; // works as expected
    k = 0; // Error: variable foo.ns.k is private
}
