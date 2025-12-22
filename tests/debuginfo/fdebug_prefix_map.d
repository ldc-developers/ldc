// RUN: %ldc -g -fdebug-prefix-map=%S=/blablabla -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// CHECK: !DIFile(filename: "/blablabla

void foo()
{
}
