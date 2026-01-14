// Tests that naked template functions work correctly with LTO linking.
// Previously, naked functions were emitted as module asm which caused
// duplicate symbol errors when the same template was instantiated in
// multiple modules and linked with LTO.
//
// The fix emits naked functions as LLVM IR functions with inline asm,
// allowing LLVM to properly deduplicate template instantiations.
//
// See: https://github.com/ldc-developers/ldc/issues/4294

// REQUIRES: LTO
// REQUIRES: target_X86

// RUN: split-file %s %t

// Compile two modules that each instantiate the same naked asm template.
// RUN: %ldc -flto=full -c -I%t %t/asm_lto_user.d -of=%t/user%obj
// RUN: %ldc -flto=full -c -I%t %t/asm_lto_main.d -of=%t/main%obj

// Link with LTO - this fails without the fix due to duplicate asm labels.
// RUN: %ldc -flto=full %t/main%obj %t/user%obj -of=%t/test%exe

// Verify the executable runs correctly.
// RUN: %t/test%exe

//--- asm_lto_template.d
// Template with naked function containing inline asm labels.
// This mimics std.internal.math.biguintx86 which triggers issue #4294.
// The naked function's asm becomes "module asm" which gets concatenated
// during LTO, causing duplicate symbol errors without the fix.
module asm_lto_template;

// Template function - when instantiated in multiple modules and linked
// with LTO, the labels must have unique IDs to avoid "symbol already defined"
uint nakedAsmTemplate(int N)() {
    version (D_InlineAsm_X86) {
        asm { naked; }
        asm {
            xor EAX, EAX;
        L1:
            add EAX, N;
            cmp EAX, 100;
            jl L1;
            ret;
        }
    } else version (D_InlineAsm_X86_64) {
        asm { naked; }
        asm {
            xor EAX, EAX;
        L1:
            add EAX, N;
            cmp EAX, 100;
            jl L1;
            ret;
        }
    } else {
        // Fallback for non-x86
        uint result = 0;
        while (result < 100) result += N;
        return result;
    }
}

//--- asm_lto_user.d
// Second module that instantiates the same naked asm template.
// This creates a separate instantiation that, when linked with LTO,
// causes "symbol already defined" errors if labels aren't unique.
module asm_lto_user;

import asm_lto_template;

uint useTemplate() {
    // Instantiate nakedAsmTemplate!1 - same as in main module
    return nakedAsmTemplate!1();
}

//--- asm_lto_main.d
module asm_lto_main;

import asm_lto_template;
import asm_lto_user;

int main() {
    // Both modules instantiate nakedAsmTemplate!1
    // Without unique label IDs, LTO linking fails with "symbol already defined"
    uint a = nakedAsmTemplate!1();  // From this module's instantiation
    uint b = useTemplate();          // From asm_lto_user's instantiation

    // Both should return the same value (>= 100)
    return (a == b && a >= 100) ? 0 : 1;
}
