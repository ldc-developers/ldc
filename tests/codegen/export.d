// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// REQUIRES: Windows

export
{
    // CHECK-DAG: define dllexport {{.*}}_D6export11exportedFooFZv
    void exportedFoo() {}

    // CHECK-DAG: declare dllimport {{.*}}_D6export11importedFooFZv
    void importedFoo();
}

void bar()
{
    exportedFoo();
    importedFoo();
}
