module tangotests.mem4;

import tango.stdc.stdio;

class C {
    int* ptr;
    this() {
        printf("this()\n");
        ptr = new int;
    }
    ~this() {
        printf("~this()\n");
        delete ptr;
        assert(ptr is null);
    }
    final void check()
    {
        printf("check()\n");
        assert(ptr !is null);
    }
}

void main()
{
    C c = new C();
    c.check();
    delete c;
    assert(c is null);
}