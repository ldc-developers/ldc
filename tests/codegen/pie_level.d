// RUN: %ldc --output-ll --output-o --relocation-model=pic -of=%t.exe %s && FileCheck %s < %t.ll

// CHECK: !{i32 8, !"PIC Level", i32 2}
// CHECK: !{i32 7, !"PIE Level", i32 2}

void main() {}
