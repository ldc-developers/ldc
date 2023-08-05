// A minimal test wrt. size_t on a 16-bit architecture.

// REQUIRES: target_AVR
// RUN: %ldc -mtriple=avr -betterC -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -mtriple=avr -betterC -c %s

static assert(size_t.sizeof == 2);
static assert(ptrdiff_t.sizeof == 2);

int testBoundsCheck(int[] arr)
{
    return arr[1];
}

// __LINE__ expressions are of type int
void takeIntLine(int line = __LINE__) {}
// special case for 8/16 bit targets: __LINE__ magically cast to size_t
void takeSizeTLine(size_t line = __LINE__) {}

void testDefaultLineArgs()
{
    // CHECK: call {{.*}}takeIntLine{{.*}}(i32 [[@LINE+1]])
    takeIntLine();
    // CHECK: call {{.*}}takeSizeTLine{{.*}}(i16 zeroext [[@LINE+1]])
    takeSizeTLine();
}
