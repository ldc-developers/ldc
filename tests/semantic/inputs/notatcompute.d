module tests.semantic.inputs.notatcompute;

void somefunc() {}

auto identity()(uint x1) => x1;

void callsSomeFunc()()
{
    somefunc();
}

struct A
{
    int foo()() { return 0; }
}

struct B()
{
    int foo() { return 0; }
}
