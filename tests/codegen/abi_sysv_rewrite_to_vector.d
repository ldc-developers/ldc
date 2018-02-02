// See https://github.com/ldc-developers/ldc/issues/2094.

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll %s && FileCheck --check-prefix=LLVM %s < %t.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -O3 -of=%t.s %s && FileCheck --check-prefix=ASM %s < %t.s

// REQUIRES: target_X86

struct Vector2f { float x, y; }
struct Vector2 { double x, y; }

// LLVM: define <2 x float> @_D26abi_sysv_rewrite_to_vector7forwardFSQBm8Vector2fZQo
Vector2f forward(Vector2f arg)
{
    // LLVM: store <2 x float>{{.*}}, align 4
    // LLVM: load <2 x float>{{.*}}, align 4
    // LLVM: ret <2 x float>
    return arg;
}

// LLVM: define <2 x double> @_D26abi_sysv_rewrite_to_vector7forwardFSQBm7Vector2ZQn
Vector2 forward(Vector2 arg)
{
    // LLVM: store <2 x double>{{.*}}, align 8
    // LLVM: load <2 x double>{{.*}}, align 8
    // LLVM: ret <2 x double>
    return arg;
}

// ASM: _D26abi_sysv_rewrite_to_vector3lenFSQBi8Vector2fZf:
float len(Vector2f p)
{
    // Make sure no GP registers are used.
    // ASM-NOT: %r
    // ASM: .cfi_endproc
    return p.x*p.x + p.y*p.y;
}
