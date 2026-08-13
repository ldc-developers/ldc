//===-- tools/runtime-adapt/source/ldcmods/sanitizers.d -----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/asan.d, sanitizer_common.d, sanitizers_optionally_linked.d
// ← driver/cl_options_sanitizers.cpp, gen/pragma.cpp LDC_extern_weak.
//
//===----------------------------------------------------------------------===//

module ldcmods.sanitizers;

import compilerparse;
import ldcmods.common;

bool wantAsan(const CompilerModel m)
{
    return hasLdcModule(m, "asan") || m.hasSanitizer;
}

bool wantSanitizerCommon(const CompilerModel m)
{
    return hasLdcModule(m, "sanitizer_common") || m.hasSanitizer;
}

bool wantSanitizersOpt(const CompilerModel m)
{
    return hasLdcModule(m, "sanitizers_optionally_linked") || m.hasSanitizer;
}

string renderAsan(const CompilerModel, string tag)
{
    return banner(tag) ~ "module ldc.asan;\n\n@system:\n@nogc:\nnothrow:\nextern (C):\n\n"
        ~ "void __asan_poison_memory_region(const(void*) addr, size_t size);\n"
        ~ "void __asan_unpoison_memory_region(const(void*) addr, size_t size);\n"
        ~ "int __asan_address_is_poisoned(const(void*) addr);\n";
}

string renderSanitizerCommon(const CompilerModel, string tag)
{
    return banner(tag) ~ "module ldc.sanitizer_common;\n\n@system:\n@nogc:\nnothrow:\nextern (C):\n\n"
        ~ "void __sanitizer_start_switch_fiber(void** fake_stack_save, const(void)* bottom, size_t size);\n"
        ~ "void __sanitizer_finish_switch_fiber(void* fake_stack_save, const(void)** bottom_old, size_t* size_old);\n";
}

string renderSanitizersOpt(const CompilerModel, string tag)
{
    return banner(tag) ~ "module ldc.sanitizers_optionally_linked;\n\n"
        ~ "// Linked when -fsanitize= is enabled (driver/cl_options_sanitizers.cpp).\n"
        ~ "version (SupportSanitizers)\n{\n"
        ~ "    version (Posix) version (OSX) {} else version = ELF;\n"
        ~ "    extern (C) @system @nogc nothrow\n    {\n"
        ~ "        version (ELF) pragma(LDC_extern_weak):\n"
        ~ "        void __sanitizer_start_switch_fiber(void** fake_stack_save, const(void)* bottom, size_t size);\n"
        ~ "        void __sanitizer_finish_switch_fiber(void* fake_stack_save, const(void)** bottom_old, size_t* size_old);\n"
        ~ "        void* __asan_get_current_fake_stack();\n"
        ~ "        void* __asan_addr_is_in_fake_stack(void* fake_stack, void* addr, void** beg, void** end);\n"
        ~ "    }\n}\n";
}
