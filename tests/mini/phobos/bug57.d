import std.stdio;
class Foo {}
void func3()
{
    Foo[1] test=[new Foo];
    writefln(test);
}
void main() {
    func3();
}
