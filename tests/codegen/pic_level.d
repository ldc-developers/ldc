// RUN: %ldc -c --output-ll --relocation-model=pic -of=%t.obj.ll %s && FileCheck %s < %t.obj.ll

// CHECK: !{i32 8, !"PIC Level", i32 2}

void main() {}
