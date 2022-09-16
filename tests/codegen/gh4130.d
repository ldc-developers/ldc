// RUN: %ldc -run %s

class Outer {
    Inner inner() {return new Inner; }
    class Inner {
        auto bar() {
            struct Range {
                void foo() {}
            }
            return Range();
        }
    }
}

void main()
{
    auto o = new Outer;
    auto i = o.inner();
    auto b = i.bar();
    b.foo();
}
