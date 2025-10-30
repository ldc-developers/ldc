// RUN: %ldc -g -fdebug-prefix-map=%S=/blablabla -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// Check that the substitution took place. Could be as `filename: "/blablabla...` or as `directory: "/blablabla...` depending on the invoke path for lit testsuite.
// CHECK: !DIFile({{.*}}: "/blablabla

void foo()
{
}
