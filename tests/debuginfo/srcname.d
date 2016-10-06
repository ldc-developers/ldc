// RUN: %ldc -g -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// CHECK-NOT: !DIFile(filename: "."

// constructs likely to produce compiler generated functions
// or that use lowering losing source file information
struct S1
{
    int x, y, z;
    C1 c; // guarantees compiler generated compare function
}

class C1
{
    int c1;
    int c2;
    C1 next;
}

enum E1
{
    value1 = 1,
}

int tryCatch(void delegate() dg)
{
    int ret;
    scope(failure) ret = 2;
    scope(exit) ret = 1;
    scope(success) ret = 3;
    try
    {
        dg();
    }
    catch(Exception e)
    {
        ret = 4;
    }
    finally
    {
        ret = 5;
    }
    return ret;
}

void forloop(int delegate(int) dg)
{
    int s = 0;
    foreach(i; 0..4)
        s += dg(i);
        
    int[] arr = new int[3];
    foreach(i; arr)
        s += dg(i);
}

void main()
{
    auto c1 = new C1;
    auto c2 = new C1;
    auto s = S1(5);
    S1 s2;
    s2 = s;
    auto e = E1.value1;
    bool scmp = s2 == s;
    bool ccmp = c1 == c2;
    
    tryCatch(() {});
    forloop((int){return 1;});
}
