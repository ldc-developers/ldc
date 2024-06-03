// REQUIRES: target_X86


// RUN: not %ldc -o- -conf=    %s 2>&1 | FileCheck --check-prefix=NO_CONF %s
// RUN: not %ldc -o- --conf "" %s 2>&1 | FileCheck --check-prefix=NO_CONF %s

// NO_CONF: Error: `object` not found. object.d may be incorrectly installed or corrupt.


// RUN: %ldc -v -o- -mtriple x86_64-vendor-windows-msvc  %s 2>&1 | FileCheck --check-prefix=TRIPLE %s
// RUN: %ldc -v -o- --mtriple=x86_64-vendor-windows-msvc %s 2>&1 | FileCheck --check-prefix=TRIPLE %s

// TRIPLE: config
// TRIPLE-SAME: (x86_64-vendor-windows-msvc)
