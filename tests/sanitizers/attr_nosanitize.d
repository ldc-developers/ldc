// Test ldc.attributes.noSanitize UDA

// RUN: %ldc -c -w -de -output-ll -fsanitize=address,thread -of=%t.normal.ll     %s -d-version=NORMAL     && FileCheck %s --check-prefix=NORMAL     < %t.normal.ll
// RUN: %ldc -c -w -de -output-ll -fsanitize=address,thread -of=%t.nosanitize.ll %s -d-version=NOSANITIZE && FileCheck %s --check-prefix=NOSANITIZE < %t.nosanitize.ll

// RUN: %ldc -wi -c -fsanitize=address -d-version=WARNING %s 2>&1 | FileCheck %s --check-prefix=WARNING

import ldc.attributes;

extern (C): // For easier name mangling

version (NORMAL)
{
    // NORMAL:      ; Function Attrs:{{.*}} sanitize_address sanitize_thread
    // NORMAL-NEXT: define{{.*}} void {{.*}}foo
    void foo()
    {
    }
}
else version (NOSANITIZE)
{
    // NOSANITIZE-NOT: sanitize_address
    // NOSANITIZE:     sanitize_thread
    // NOSANITIZE-NOT: sanitize_address
    // NOSANITIZE:     define{{.*}} void {{.*}}foo_noaddress
    @noSanitize("address")
    void foo_noaddress()
    {
    }

    // NOSANITIZE-NOT: ; Function Attrs:{{.*}} sanitize_
    // NOSANITIZE:     define{{.*}} void {{.*}}foo_nothread_noaddress
    @noSanitize("thread")
    @noSanitize("address")
    void foo_nothread_noaddress()
    {
    }
}
else version (WARNING)
{
    // WARNING: attr_nosanitize.d([[@LINE+1]]){{.*}} unrecognized sanitizer name 'invalid_name'
    @noSanitize("invalid_name")
    void foo_error()
    {
    }
}
