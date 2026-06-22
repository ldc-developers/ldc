// No-regression guard (Cat 4): the comparison/equality-hook allowance must NOT
// drag CTFE-only host templates into device legality checking.
//
// std.traits.fullyQualifiedName / moduleName are host-only templates whose
// implementations use array concatenation (`~`) -- lowered to host runtime
// helpers in core.internal.array.concatenation / .utils that do GC.malloc/memcpy
// and are genuinely device-illegal. Used purely at compile time (here: to derive
// CT lengths) they must not trip any device error, because the concat helpers run
// only during CTFE and never reach device codegen. The hook allowance is scoped to
// core.internal.array.comparison / .equality, so it must leave these alone.
//
// This is the exact case that regressed under an over-broad "any instantiated
// template is device-legal" predicate; it must compile cleanly (exit 0).
//
// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-700 %s

@compute(CompileFor.deviceOnly) module dcompute_ctfe_host_templates;
import ldc.dcompute;
import std.traits : fullyQualifiedName, moduleName;

struct SomeType { int x; }

@kernel void k(GlobalPointer!int o)
{
    // CT-only consumption of the host template results -- never materializes the
    // string at runtime, so no host array machinery reaches device codegen.
    enum nFqnInt   = fullyQualifiedName!int.length;
    enum nFqnType  = fullyQualifiedName!SomeType.length;
    enum nModName  = moduleName!SomeType.length;

    int acc = 0;
    static foreach (i; 0 .. nFqnInt)
        acc += i;

    *o = acc + cast(int)(nFqnType + nModName);
}
