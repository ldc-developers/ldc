// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// REQUIRES: Windows

export
{
    // CHECK: @{{.*}}exportedGlobal{{.*}} = dllexport
    __gshared void* exportedGlobal;

    // TLS: unsupported => linker errors
    // CHECK: @{{.*}}exportedTlsGlobal{{.*}} = thread_local
    // CHECK-NOT: dllexport
    void* exportedTlsGlobal;

    extern
    {
        // CHECK: @{{.*}}importedGlobal{{.*}} = external dllimport
        __gshared void* importedGlobal;

        // CHECK: @{{.*}}importedTlsGlobal{{.*}} = external thread_local
        // CHECK-NOT: dllimport
        void* importedTlsGlobal;
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
    // make sure the imported symbols are IR-declared
    importedGlobal = null;
    importedTlsGlobal = null;
    importedFoo();
}
