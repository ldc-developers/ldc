import std.stdio;

enum E : string
{
    A = "A",
    B = "B",
    C = "C",
}

void main(string[] args)
{
    string[E] aa = [E.A : "a", E.B : "b", E.C : "c"];
    assert(aa[E.A] == "a");
    assert(aa[E.B] == "b");
    assert(aa[E.C] == "c");
}

