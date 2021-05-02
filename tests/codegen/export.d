// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// REQUIRES: Windows

export
{
    // CHECK-DAG: @{{.*}}exportedGlobal{{.*}} = dllexport
    extern(C) __gshared void* exportedGlobal;

    // CHECK-DAG: @{{.*}}importedGlobal{{.*}} = external dllimport
    extern(C) extern __gshared void* importedGlobal;

    // CHECK-DAG: define{{( dso_local)?}} dllexport {{.*}}_D6export11exportedFooFZv
    void exportedFoo() {}

    // CHECK-DAG: declare dllimport {{.*}}_D6export11importedFooFZv
    void importedFoo();
}

void bar()
{
    exportedFoo();
    importedFoo();
}
