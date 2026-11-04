// Thread-local definitions are dso_local if they cannot be preempted (non-PIC
// code, or linking an executable ourselves), which enables the local-exec TLS
// model. Declarations are not, as they may be defined in a shared library.

// REQUIRES: Linux

// RUN: %ldc -c --relocation-model=pic -of=%t.def.o %S/inputs/tls_dso_local_def.d

// RUN: %ldc --output-ll --output-o --relocation-model=pic -I%S %s %t.def.o -of=%t.exe && FileCheck %s --check-prefixes=COMMON,EXE < %t.ll
// RUN: %ldc -c --output-ll --relocation-model=pic -I%S %s -of=%t.obj.ll && FileCheck %s --check-prefixes=COMMON,NOEXE < %t.obj.ll
// RUN: %ldc -shared --output-ll --output-o -I%S %s %t.def.o -of=%t.so && FileCheck %s --check-prefixes=COMMON,NOEXE < %t.ll

import inputs.tls_dso_local_def;

// EXE-DAG: @_D13tls_dso_local3tlsi = dso_local thread_local
// NOEXE-DAG: @_D13tls_dso_local3tlsi = thread_local
int tls;

// EXE-DAG: @_D13tls_dso_local__T4TmplTiZ1xi = weak_odr dso_local thread_local
// NOEXE-DAG: @_D13tls_dso_local__T4TmplTiZ1xi = weak_odr thread_local
template Tmpl(T) { T x; }

// COMMON-DAG: @_D13tls_dso_local7gsharedi = global
__gshared int gshared;

// COMMON-DAG: @_D6inputs17tls_dso_local_def11importedTLSi = external thread_local global
void main() {
    tls = importedTLS + Tmpl!int.x + gshared;
}
