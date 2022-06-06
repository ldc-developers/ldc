// RUN: %ldc -run %s

template listDir(alias handler)
{
    struct NestedStruct
    {
        void callHandler() { handler(); }
    }

    class NestedClass
    {
        void callHandler() { handler(); }
    }

    void nestedFunc() { handler(); }

    void listDir()
    {
        int a = 123;
        void foo() { assert(a == 123); }

        // pass local listDir() frame as context
        foo();

        // pass parent context for sibling symbols:
        NestedStruct().callHandler();
        (new NestedClass).callHandler();
        nestedFunc();
    }
}

void main()
{
    int magic = 0xDEADBEEF;
    listDir!(() { assert(magic == 0xDEADBEEF); })();
}
