// On Windows, re-test dmd-testsuite's runnable\eh.d with all 4 MS C runtime variants.
// In case of failures, it's very likely that the EH terminate hook in druntime's
// ldc.eh.msvc.msvc_eh_terminate() needs to be adapted.

// REQUIRES: Windows

// RUN: %ldc -mscrtlib=libcmt                         -run "%S\..\d2\dmd-testsuite\runnable\eh.d"
// RUN: %ldc -mscrtlib=libcmtd -link-defaultlib-debug -run "%S\..\d2\dmd-testsuite\runnable\eh.d"
// RUN: %ldc -mscrtlib=msvcrt  -link-defaultlib-debug -run "%S\..\d2\dmd-testsuite\runnable\eh.d"
// RUN: %ldc -mscrtlib=msvcrtd                        -run "%S\..\d2\dmd-testsuite\runnable\eh.d"
