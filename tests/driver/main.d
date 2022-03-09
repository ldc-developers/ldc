// compile & link at once with -main:
// RUN: %ldc -main %s %S/inputs/include_imports2.d -of=%t1%exe
// RUN: %t1%exe | FileCheck %s

// compile separately, then link with -main:
// RUN: %ldc -c %s %S/inputs/include_imports2.d -od=%t.obj
// RUN: %ldc -main %t.obj/main%obj %t.obj/include_imports2%obj -of=%t2%exe
// RUN: %t2%exe | FileCheck %s

module main_;

import core.stdc.stdio : printf;
import inputs.include_imports2 : bar;

shared static this()
{
    bar();
    // CHECK: module ctor
    printf("module ctor\n");
}
