/* Make sure -platformlib overrides the default platform libraries list.
 * We only care about the platform libs in the linker command line;
 * make sure linking fails in all cases (no main()) as linking would
 * fail without the platform libraries anyway. Finally this option is
 * relevant only for windows targets so make sure we target Windows with
 * -mtriple=x86_64-unknown-windows-coff.
 */

// RUN: not %ldc -v -mtriple=x86_64-unknown-windows-coff -platformlib= %s | FileCheck %s
// CHECK-NOT: kernel32.lib
// CHECK-NOT: user32
// CHECK-NOT: gdi32
// CHECK-NOT: winspool
// CHECK-NOT: shell32
// CHECK-NOT: ole32
// CHECK-NOT: oleaut32
// CHECK-NOT: uuid
// CHECK-NOT: comdlg32
// CHECK-NOT: advapi32
// CHECK-NOT: oldnames
// CHECK-NOT: legacy_stdio_definitions
