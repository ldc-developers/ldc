// Make sure -platformlib overrides the default platform libraries list.


// RUN: %ldc %s -platformlib= -gcc=echo -linker=echo | FileCheck --check-prefix=EMPTY %s

// EMPTY-NOT: -lrt
// EMPTY-NOT: -ldl
// EMPTY-NOT: -lpthread
// EMPTY-NOT: -lm

// EMPTY-NOT: kernel32
// EMPTY-NOT: user32
// EMPTY-NOT: gdi32
// EMPTY-NOT: winspool
// EMPTY-NOT: shell32
// EMPTY-NOT: ole32
// EMPTY-NOT: oleaut32
// EMPTY-NOT: uuid
// EMPTY-NOT: comdlg32
// EMPTY-NOT: advapi32
// EMPTY-NOT: oldnames
// EMPTY-NOT: legacy_stdio_definitions


// RUN: %ldc %s -platformlib=myPlatformLib1,myPlatformLib2 -gcc=echo -linker=echo | FileCheck --check-prefix=CUSTOM %s

// CUSTOM: {{(-lmyPlatformLib1 -lmyPlatformLib2)|(myPlatformLib1.lib myPlatformLib2.lib)}}
