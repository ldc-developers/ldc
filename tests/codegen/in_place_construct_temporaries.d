// from https://issues.dlang.org/show_bug.cgi?id=20321

// Restrict to x86[_64] hosts for now, as the ABI must not perform any implicit
// blits (e.g., via LLVM byval attribute) for non-PODs.
// REQUIRES: host_X86
// RUN: %ldc -run %s

version (Win32)
{
    // ABI needs *a lot* of work: https://github.com/ldc-developers/ldc/pull/3204#discussion_r339300174
    void main() {}
}
else:

__gshared bool success = true;

/** Container with internal pointer
 */
struct Container
{
    long[3] data;
    void* p;

    this(int) { p = &data[0]; }
    this(ref inout Container) inout { p = &data[0]; }

    /** Ensure the internal pointer is correct */
    void check(int line = __LINE__)
    {
        if (p != &data[0])
        {
            import core.stdc.stdio : printf;
            printf("Check failed in line %d\n", line);
            success = false;
        }
    }
}

void func(Container c) { c.check(); } // error

Container get()
{
    auto a = Container(1);
    auto b = a;
    a.check(); // ok
    b.check(); // ok
    // no nrvo
    if (1)
        return a;
    else
        return b;
}

Container get2()
out(r){}
do
{
    auto v = Container(1);
    v.check(); // ok
    return v;
}

int main()
{
    Container v = Container(1);
    v.check(); // ok

    func(v);
    auto r = get();
    r.check(); // error

    auto r2 = get2();
    r.check(); // error

    Container[1] slit = [v];
    slit[0].check(); // error

    Container[] dlit = [v];
    dlit[0].check(); // error

    auto b = B(v);
    b.m.check(); // error

    return success ? 0 : 1;
}

struct B
{
    Container m;
}
