// RUN: %ldc -run %s && %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

enum offset = 0xFFFF_FFFF_0000_0000UL;
void main() {
    // CHECK: %1 = getelementptr inbounds i8,{{.*}}_Dmain{{.*}}, i64 -4294967296
    assert((cast(ulong)&main) != (cast(ulong)&main + offset));
}
