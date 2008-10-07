extern(C) int printf(char*, ...);
template Foo() { void test() { printf("test\n"); typeof(this).whee(); } }
class Bar { void whee() { printf("whee\n"); } mixin Foo!(); }
void main() { (new Bar).test(); }
