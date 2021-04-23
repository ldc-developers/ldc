// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// REQUIRES: Windows

export
{
    // CHECK: @{{.*}}exportedGlobal{{.*}} = dllexport
    extern(C) __gshared void* exportedGlobal;

    // CHECK: @{{.*}}importedGlobal{{.*}} = external dllimport
    extern(C) extern __gshared void* importedGlobal;

    // CHECK: define dllexport {{.*}}_D6export11exportedFooFZv
    void exportedFoo() {}

    // CHECK: declare
    // CHECK-NOT: dllimport
    // CHECK-SAME: _D6export11importedFooFZv
    void importedFoo();
}

void bar()
{
    exportedFoo();
    importedFoo();
}
