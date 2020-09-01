// Tests that static arrays can be large (> 16MB)

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: Stuff = type { [209715200 x i8] }
struct Stuff
{
    byte[1024*1024*200] a;
}
Stuff stuff;

// CHECK: hugeArrayG209715200g{{\"?}} ={{.*}} [209715200 x i8]
byte[1024*1024*200] hugeArray;
