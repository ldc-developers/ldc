struct Foo(E1, E2) {
    Spam tm;
    static struct Bar {
        this(in E2, in E1) {}
    }
    static struct Spam {
        Bar[E2][E1] bars;
    }
}
void main() {
    import std.stdio: writeln;
    writeln("hello world");
    enum E3 { A, B }
    enum E4 { C, D }
    alias M1 = Foo!(E3, E4);
    M1.Spam s;
    s.bars = [E3.A: [E4.C: M1.Bar(E4.D, E3.B)]];
}