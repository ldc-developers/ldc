// Test timetrace2txt tool basic TSV functionality

// RUN: %ldc -c -o- --ftime-trace --ftime-trace-file=%t.timetrace --ftime-trace-granularity=1 %s

// RUN: %timetrace2txt %t.timetrace -o %t.txt --tsv %t.tsv && FileCheck %s < %t.tsv
// RUN: %timetrace2txt %t.timetrace -o %t.txt --tsv - | FileCheck %s

// CHECK: Duration Text Line Number Name Location Detail

void foo();

void main()
{
    foo();
}
