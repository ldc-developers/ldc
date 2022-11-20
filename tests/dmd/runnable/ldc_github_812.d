import std.stdio;

interface A {}

class B : A {
    ubyte[0] test;
}

void main() {
    A a = new B();
    B b = cast(B) a;
}