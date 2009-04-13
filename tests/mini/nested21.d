module nested21;

extern(C) int printf(char*, ...);

void main() {
    int i = 42;
    int foo() { return i; }
    int bar() {
        int j = 47;
        int baz() { return j; }
        return foo() + baz();
    }
    auto result = bar();
    printf("%d\n", result);
    assert(result == 42 + 47);
}
