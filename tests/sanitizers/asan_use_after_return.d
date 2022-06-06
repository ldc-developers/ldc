// REQUIRES: ASan, RTSupportsSanitizers

// Note on debug lineinfo: on macOS the executable contains a link back to the
// object files for debug info. Therefore the order of text execution is important,
// i.e. we should finish all testing on one compiled executable before recompiling
// with different conditional compilation settings (because it will overwrite the
// object files from the previous compilation).

// RUN: %ldc -g -fsanitize=address %s -of=%t%exe
// RUN: not env %env_asan_opts=detect_stack_use_after_return=true %t%exe 2>&1 | FileCheck %s

import core.memory;
import std.stdio;

// CHECK: ERROR: AddressSanitizer: stack-use-after-return
// CHECK-NEXT: READ of size 4

struct S(Dlg) { Dlg dlg; }
auto invoker(Dlg)(scope Dlg dlg) { return S!Dlg(dlg); }
@nogc auto f(int x) {
    scope dlg = delegate() {
// CHECK-NEXT: #0 {{.*}} in {{.*}}asan_use_after_return.d:[[@LINE+1]]
                    x++;
                };
    return invoker(dlg);
}

void main()
{
    auto inv = f(2);
// CHECK-NEXT: #1 {{.*}} in {{.*}}asan_use_after_return.d
    inv.dlg();
}

// CHECK: Address {{.*}} is located in stack of
// CHECK-NEXT: #0 {{.*}} in {{.*}}asan_use_after_return.d:[[@LINE-16]]
