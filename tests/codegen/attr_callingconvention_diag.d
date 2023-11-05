// Test diagnostics for unrecognized ldc.attributes.callingConvention

// RUN: not %ldc -w -c %s 2>&1 | FileCheck %s

import ldc.attributes;

// CHECK: attr_callingconvention_diag.d([[@LINE+1]]): Warning: ignoring unrecognized calling convention name 'bogus calling convention' for `@ldc.attributes.callingConvention`
@callingConvention("bogus calling convention")
void foofoofoo()
{
}
