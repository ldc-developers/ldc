//===-- tools/runtime-adapt/source/ldcmods/eh_wasm.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/eh_wasm.d — added in 28933e1577 ("Add LDC-specific DRuntime changes"),
// later folded into rt/dwarfeh.d by 21ba72d91f ("Add support for Wasm EH").
// Emit only when this tag's compiler still names ldc.eh_wasm.
//
//===----------------------------------------------------------------------===//

module ldcmods.eh_wasm;

import compilerparse;
import ldcmods.common;

bool wantEhWasm(const CompilerModel m)
{
    return hasLdcModule(m, "eh_wasm");
}

string renderEhWasm(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
module ldc.eh_wasm;

// Wasm EH personality / catch enter. Compiler: gen/trycatchfinally.cpp,
// gen/runtime.cpp. Implementation may live in rt/dwarfeh.d on later tags.
extern (C) int _d_eh_personality();
extern (C) Throwable _d_eh_enter_catch(void* exception);
extern (C) void _d_eh_resume_unwind(void* ptr);
D";
}
