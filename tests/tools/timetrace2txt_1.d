// Test timetrace2txt tool basic functionality

// RUN: %ldc -c -o- --ftime-trace --ftime-trace-file=%t.timetrace --ftime-trace-granularity=1 %s

// RUN: %timetrace2txt %t.timetrace -o %t.txt && FileCheck %s < %t.txt
// RUN: %timetrace2txt %t.timetrace -o - | FileCheck %s

// CHECK: Timetrace
// CHECK: main

void foo();

void main()
{
    foo();
}
