module bug1;
struct Foo { Foo opSub(ref Foo b) { return Foo(); } }
struct Bar { Foo whee; }
void test(ref Bar moo) { Foo nngh; auto plonk = nngh - moo.whee; }
void main() { Bar bar; test(bar); }
