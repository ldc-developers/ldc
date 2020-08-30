module gh3548a;

struct S {
    pragma(inline, true) auto innerStruct() {
        static struct InnerStruct {
            int nonZero = 1;
        }
        return InnerStruct();
    }
    pragma(inline, true) auto innerInnerStruct() {
        struct InnerStruct {
            struct InnerInnerStruct {
                int nonZero = 1;
            }
            int nonZero = 1;
        }
        return InnerStruct.InnerInnerStruct();
    }
    pragma(inline, true) auto innerClass() {
        interface I { void foo(); }
        class InnerClass : I {
            void foo() {}
            int nonZero = 1;
        }
        return new InnerClass();
    }
    pragma(inline, true) auto innerInnerClass() {
        static class InnerClass {
            static interface I { void foo(); }
            static class InnerInnerClass : I {
                void foo() {}
                int nonZero = 1;
            }
            int nonZero = 1;
        }
        return new InnerClass.InnerInnerClass();
    }
}

//Not needed but simply making sure we generate code for the types.
auto fooInnerStruct() {
    S s;
    return s.innerStruct();
}
auto fooInnerInnerStruct() {
    S s;
    return s.innerInnerStruct();
}
auto fooInnerClass() {
    S s;
    return s.innerClass();
}
auto fooInnerInnerClass() {
    S s;
    return s.innerInnerClass();
}
