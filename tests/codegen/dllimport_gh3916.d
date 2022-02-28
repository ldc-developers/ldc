// REQUIRES: Windows

// RUN: %ldc -output-ll -dllimport=all             -of=%t_all.ll %s && FileCheck %s < %t_all.ll
// RUN: %ldc -output-ll -dllimport=defaultLibsOnly -of=%t_dlo.ll %s && FileCheck %s < %t_dlo.ll

import std.random : Xorshift; // pre-instantiated template

void foo()
{
    // CHECK: _D3std6random__T14XorshiftEngine{{.*}}6__initZ = external dllimport
    const i = __traits(initSymbol, Xorshift);
}
