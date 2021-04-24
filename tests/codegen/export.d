// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// REQUIRES: Windows

export
{
    // non-TLS:
    __gshared
    {
        // CHECK: @{{.*}}exportedGlobal{{.*}} = dllexport
        void* exportedGlobal;

        // CHECK: @{{.*}}importedGlobal{{.*}} = external dllimport
        extern void* importedGlobal;
    }

    // TLS: unsupported => linker errors
    version (all)
    {
        // CHECK: @{{.*}}exportedTlsGlobal{{.*}} = thread_local
        // CHECK-NOT: dllexport
        void* exportedTlsGlobal;

        // CHECK: @{{.*}}importedTlsGlobal{{.*}} = external thread_local
        // CHECK-NOT: dllimport
        extern void* importedTlsGlobal;
    }

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
