// RUN: %ldc -shared -of=%t.dll %s
// RUN: dumpbin /exports %t.dll | FileCheck %s

// REQUIRES: Windows

export
{
    // CHECK-DAG: _D19export_naked_gh264812exportNormal
    void exportNormal() { }
    // CHECK-DAG: _D19export_naked_gh264811exportNaked
    void exportNaked() { asm { naked; } }
}
