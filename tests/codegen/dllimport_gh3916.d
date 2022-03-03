// REQUIRES: Windows

// RUN: %ldc -output-ll -dllimport=all             -of=%t_all.ll %s && FileCheck %s --check-prefix=ALL < %t_all.ll
// RUN: %ldc -output-ll -dllimport=defaultLibsOnly -of=%t_dlo.ll %s && FileCheck %s --check-prefix=DLO < %t_dlo.ll

import std.random : Xorshift; // pre-instantiated template

void foo()
{
    const i = __traits(initSymbol, Xorshift);
}

// -dllimport=all: dllimport declaration
// ALL: @_D3std6random__T14XorshiftEngine{{.*}}6__initZ = external dllimport constant

// -dllimport=defaultLibsOnly: define-on-declare instantiated druntime/Phobos data symbols
// see https://github.com/ldc-developers/ldc/issues/3931
// DLO: @_D3std6random__T14XorshiftEngine{{.*}}6__initZ = weak_odr constant
