/*
 * This module holds declarations to LLVM intrinsics.
 *
 * See the LLVM language reference for more information:
 *
 * - http://llvm.org/docs/LangRef.html#intrinsics
 *
 */

module ldc.intrinsics;

// Check for the right compiler
version(LDC)
{
    // OK
}
else
{
    static assert(false, "This module is only valid for LDC");
}

version(LDC_LLVM_302) version = INTRINSICS_FROM_302;
version(LDC_LLVM_303)
{
    version = INTRINSICS_FROM_302;
    version = INTRINSICS_FROM_303;
}
version(LDC_LLVM_304)
{
    version = INTRINSICS_FROM_302;
    version = INTRINSICS_FROM_303;
    version = INTRINSICS_FROM_304;
}
version(LDC_LLVM_305)
{
    version = INTRINSICS_FROM_302;
    version = INTRINSICS_FROM_303;
    version = INTRINSICS_FROM_304;
    version = INTRINSICS_FROM_305;
}
version(LDC_LLVM_306)
{
    version = INTRINSICS_FROM_302;
    version = INTRINSICS_FROM_303;
    version = INTRINSICS_FROM_304;
    version = INTRINSICS_FROM_305;
    version = INTRINSICS_FROM_306;
}
version(LDC_LLVM_307)
{
    version = INTRINSICS_FROM_302;
    version = INTRINSICS_FROM_303;
    version = INTRINSICS_FROM_304;
    version = INTRINSICS_FROM_305;
    version = INTRINSICS_FROM_306;
    version = INTRINSICS_FROM_307;
}

// All intrinsics are nothrow and @nogc. The codegen intrinsics are not categorized
// any further (they probably could), the rest is pure (aborting is fine by
// definition; memcpy and friends can be viewed as weakly pure, just as e.g.
// strlen() is marked weakly pure as well) and mostly @safe.
nothrow:
@nogc:

//
// CODE GENERATOR INTRINSICS
//


/// The 'llvm.returnaddress' intrinsic attempts to compute a target-specific
/// value indicating the return address of the current function or one of its
/// callers.

pragma(LDC_intrinsic, "llvm.returnaddress")
    void* llvm_returnaddress(uint level);


/// The 'llvm.frameaddress' intrinsic attempts to return the target-specific
/// frame pointer value for the specified stack frame.

pragma(LDC_intrinsic, "llvm.frameaddress")
    void* llvm_frameaddress(uint level);


/// The 'llvm.stacksave' intrinsic is used to remember the current state of the
/// function stack, for use with llvm.stackrestore. This is useful for
/// implementing language features like scoped automatic variable sized arrays
/// in C99.

pragma(LDC_intrinsic, "llvm.stacksave")
    void* llvm_stacksave();


/// The 'llvm.stackrestore' intrinsic is used to restore the state of the
/// function stack to the state it was in when the corresponding llvm.stacksave
/// intrinsic executed. This is useful for implementing language features like
/// scoped automatic variable sized arrays in C99.

pragma(LDC_intrinsic, "llvm.stackrestore")
    void llvm_stackrestore(void* ptr);


/// The 'llvm.prefetch' intrinsic is a hint to the code generator to insert a
/// prefetch instruction if supported; otherwise, it is a noop. Prefetches have
/// no effect on the behavior of the program but can change its performance
/// characteristics.
/// ptr is the address to be prefetched, rw is the specifier determining if the
/// fetch should be for a read (0) or write (1), and locality is a temporal
/// locality specifier ranging from (0) - no locality, to (3) - extremely local
/// keep in cache. The cache type specifies whether the prefetch is performed on
/// the data (1) or instruction (0) cache. The rw, locality and cache type
/// arguments must be constant integers.

pragma(LDC_intrinsic, "llvm.prefetch")
    void llvm_prefetch(void* ptr, uint rw, uint locality, uint cachetype);


/// The 'llvm.pcmarker' intrinsic is a method to export a Program Counter (PC)
/// in a region of code to simulators and other tools. The method is target
/// specific, but it is expected that the marker will use exported symbols to
/// transmit the PC of the marker. The marker makes no guarantees that it will
/// remain with any specific instruction after optimizations. It is possible
/// that the presence of a marker will inhibit optimizations. The intended use
/// is to be inserted after optimizations to allow correlations of simulation
/// runs.

pragma(LDC_intrinsic, "llvm.pcmarker")
    void llvm_pcmarker(uint id);


/// The 'llvm.readcyclecounter' intrinsic provides access to the cycle counter
/// register (or similar low latency, high accuracy clocks) on those targets that
/// support it. On X86, it should map to RDTSC. On Alpha, it should map to RPCC.
/// As the backing counters overflow quickly (on the order of 9 seconds on
/// alpha), this should only be used for small timings.

pragma(LDC_intrinsic, "llvm.readcyclecounter")
    ulong llvm_readcyclecounter();

// Backwards compatibility - but it is doubtful whether somebody actually ever
// used that intrinsic.
alias llvm_readcyclecounter readcyclecounter;


version(INTRINSICS_FROM_305)
{
/// The 'llvm.clear_cache' intrinsic ensures visibility of modifications in the
/// specified range to the execution unit of the processor. On targets with
/// non-unified instruction and data cache, the implementation flushes the
/// instruction cache.
/// On platforms with coherent instruction and data caches (e.g. x86), this
/// intrinsic is a nop. On platforms with non-coherent instruction and data
/// cache (e.g. ARM, MIPS), the intrinsic is lowered either to appropriate
/// instructions or a system call, if cache flushing requires special privileges.
///
/// The default behavior is to emit a call to __clear_cache from the run time library.
///
/// This instrinsic does not empty the instruction pipeline. Modifications of
/// the current function are outside the scope of the intrinsic.
pragma(LDC_intrinsic, "llvm.clear_cache")
    void llvm_clear_cache(void *from, void *to);
}



//
// STANDARD C LIBRARY INTRINSICS
//


pure:

/// The 'llvm.memcpy.*' intrinsics copy a block of memory from the source
/// location to the destination location.
/// Note that, unlike the standard libc function, the llvm.memcpy.* intrinsics do
/// not return a value, and takes an extra alignment argument.

pragma(LDC_intrinsic, "llvm.memcpy.p0i8.p0i8.i#")
    void llvm_memcpy(T)(void* dst, void* src, T len, uint alignment, bool volatile_ = false);


/// The 'llvm.memmove.*' intrinsics move a block of memory from the source
/// location to the destination location. It is similar to the 'llvm.memcpy'
/// intrinsic but allows the two memory locations to overlap.
/// Note that, unlike the standard libc function, the llvm.memmove.* intrinsics
/// do not return a value, and takes an extra alignment argument.

pragma(LDC_intrinsic, "llvm.memmove.p0i8.p0i8.i#")
    void llvm_memmove(T)(void* dst, void* src, T len, uint alignment, bool volatile_ = false);


/// The 'llvm.memset.*' intrinsics fill a block of memory with a particular byte
/// value.
/// Note that, unlike the standard libc function, the llvm.memset intrinsic does
/// not return a value, and takes an extra alignment argument.

pragma(LDC_intrinsic, "llvm.memset.p0i8.i#")
    void llvm_memset(T)(void* dst, ubyte val, T len, uint alignment, bool volatile_ = false);


@safe:

/// The 'llvm.sqrt' intrinsics return the sqrt of the specified operand,
/// returning the same value as the libm 'sqrt' functions would. Unlike sqrt in
/// libm, however, llvm.sqrt has undefined behavior for negative numbers other
/// than -0.0 (which allows for better optimization, because there is no need to
/// worry about errno being set). llvm.sqrt(-0.0) is defined to return -0.0 like
/// IEEE sqrt.

pragma(LDC_intrinsic, "llvm.sqrt.f#")
    T llvm_sqrt(T)(T val);


/// The 'llvm.sin.*' intrinsics return the sine of the operand.

pragma(LDC_intrinsic, "llvm.sin.f#")
    T llvm_sin(T)(T val);


/// The 'llvm.cos.*' intrinsics return the cosine of the operand.

pragma(LDC_intrinsic, "llvm.cos.f#")
    T llvm_cos(T)(T val);


/// The 'llvm.powi.*' intrinsics return the first operand raised to the specified
/// (positive or negative) power. The order of evaluation of multiplications is
/// not defined. When a vector of floating point type is used, the second
/// argument remains a scalar integer value.

pragma(LDC_intrinsic, "llvm.powi.f#")
    T llvm_powi(T)(T val, int power);


/// The 'llvm.pow.*' intrinsics return the first operand raised to the specified
/// (positive or negative) power.

pragma(LDC_intrinsic, "llvm.pow.f#")
    T llvm_pow(T)(T val, T power);


/// The 'llvm.exp.*' intrinsics perform the exp function.

pragma(LDC_intrinsic, "llvm.exp.f#")
    T llvm_exp(T)(T val);


/// The 'llvm.log.*' intrinsics perform the log function.

pragma(LDC_intrinsic, "llvm.log.f#")
    T llvm_log(T)(T val);


/// The 'llvm.fma.*' intrinsics perform the fused multiply-add operation.

pragma(LDC_intrinsic, "llvm.fma.f#")
    T llvm_fma(T)(T vala, T valb, T valc);

version(INTRINSICS_FROM_302)
{
/// The 'llvm.fabs.*' intrinsics return the absolute value of the operand.

pragma(LDC_intrinsic, "llvm.fabs.f#")
    T llvm_fabs(T)(T val);


/// The 'llvm.floor.*' intrinsics return the floor of the operand.

pragma(LDC_intrinsic, "llvm.floor.f#")
    T llvm_floor(T)(T val);
}

version(INTRINSICS_FROM_303)
{
/// The 'llvm.exp2.*' intrinsics perform the exp2 function.

pragma(LDC_intrinsic, "llvm.exp2.f#")
    T llvm_exp2(T)(T val);

/// The 'llvm.log10.*' intrinsics perform the log10 function.

pragma(LDC_intrinsic, "llvm.log10.f#")
    T llvm_log10(T)(T val);

/// The 'llvm.log2.*' intrinsics perform the log2 function.

pragma(LDC_intrinsic, "llvm.log2.f#")
    T llvm_log2(T)(T val);

/// The 'llvm.ceil.*' intrinsics return the ceiling of the operand.

pragma(LDC_intrinsic, "llvm.ceil.f#")
    T llvm_ceil(T)(T val);

/// The 'llvm.trunc.*' intrinsics returns the operand rounded to the nearest integer not larger in magnitude than the operand.

pragma(LDC_intrinsic, "llvm.trunc.f#")
    T llvm_trunc(T)(T val);

/// The 'llvm.rint.*' intrinsics returns the operand rounded to the nearest integer. It may raise an inexact floating-point exception if the operand isn't an integer.

pragma(LDC_intrinsic, "llvm.rint.f#")
    T llvm_rint(T)(T val);

/// The 'llvm.nearbyint.*' intrinsics returns the operand rounded to the nearest integer.

pragma(LDC_intrinsic, "llvm.nearbyint.f#")
    T llvm_nearbyint(T)(T val);
}

version(INTRINSICS_FROM_304)
{

/// The 'llvm.copysign.*' intrinsics return a value with the magnitude of the first operand and the sign of the second operand.

pragma(LDC_intrinsic, "llvm.copysign.f#")
    T llvm_copysign(T)(T mag, T sgn);

/// The 'llvm.round.*' intrinsics returns the operand rounded to the nearest integer.

pragma(LDC_intrinsic, "llvm.round.f#")
    T llvm_round(T)(T val);

}

/// The 'llvm.fmuladd.*' intrinsic functions represent multiply-add expressions
/// that can be fused if the code generator determines that the fused expression
///  would be legal and efficient.

pragma(LDC_intrinsic, "llvm.fmuladd.f#")
    T llvm_fmuladd(T)(T vala, T valb, T valc);


version(INTRINSICS_FROM_306)
{
/// The ‘llvm.minnum.*‘ intrinsics return the minimum of the two arguments.
/// Follows the IEEE-754 semantics for minNum, which also match for libm’s fmin.
/// If either operand is a NaN, returns the other non-NaN operand. Returns NaN
/// only if both operands are NaN. If the operands compare equal, returns a value
/// that compares equal to both operands. This means that fmin(+/-0.0, +/-0.0)
/// could return either -0.0 or 0.0.

pragma(LDC_intrinsic, "llvm.minnum.f#")
	T llvm_minnum(T)(T vala, T valb);

/// The ‘llvm.maxnum.*‘ intrinsics return the maximum of the two arguments.
/// Follows the IEEE-754 semantics for maxNum, which also match for libm’s fmax.
/// If either operand is a NaN, returns the other non-NaN operand. Returns NaN
/// only if both operands are NaN. If the operands compare equal, returns a value
/// that compares equal to both operands. This means that fmax(+/-0.0, +/-0.0)
/// could return either -0.0 or 0.0.

pragma(LDC_intrinsic, "llvm.maxnum.f#")
	T llvm_maxnum(T)(T vala, T valb);
}

//
// BIT MANIPULATION INTRINSICS
//

/// The 'llvm.bswap' family of intrinsics is used to byte swap integer values
/// with an even number of bytes (positive multiple of 16 bits). These are
/// useful for performing operations on data that is not in the target's native
/// byte order.

pragma(LDC_intrinsic, "llvm.bswap.i#")
    T llvm_bswap(T)(T val);


/// The 'llvm.ctpop' family of intrinsics counts the number of bits set in a
/// value.

pragma(LDC_intrinsic, "llvm.ctpop.i#")
    T llvm_ctpop(T)(T src);


/// The 'llvm.ctlz' family of intrinsic functions counts the number of leading
/// zeros in a variable.

pragma(LDC_intrinsic, "llvm.ctlz.i#")
    T llvm_ctlz(T)(T src, bool isZerodefined);


/// The 'llvm.cttz' family of intrinsic functions counts the number of trailing
/// zeros.

pragma(LDC_intrinsic, "llvm.cttz.i#")
    T llvm_cttz(T)(T src, bool isZerodefined);


//
// ATOMIC OPERATIONS AND SYNCHRONIZATION INTRINSICS
//

enum AtomicOrdering {
  NotAtomic = 0,
  Unordered = 1,
  Monotonic = 2,
  Consume = 3,
  Acquire = 4,
  Release = 5,
  AcquireRelease = 6,
  SequentiallyConsistent = 7
};
alias AtomicOrdering.SequentiallyConsistent DefaultOrdering;

/// The 'fence' intrinsic is used to introduce happens-before edges between operations.
pragma(LDC_fence)
    void llvm_memory_fence(AtomicOrdering ordering = DefaultOrdering);

/// This intrinsic loads a value stored in memory at ptr.
pragma(LDC_atomic_load)
    T llvm_atomic_load(T)(in shared T *ptr, AtomicOrdering ordering = DefaultOrdering);

/// This intrinsic stores a value in val in the memory at ptr.
pragma(LDC_atomic_store)
    void llvm_atomic_store(T)(T val, shared T *ptr, AtomicOrdering ordering = DefaultOrdering);


/// This loads a value in memory and compares it to a given value. If they are
/// equal, it stores a new value into the memory.

pragma(LDC_atomic_cmp_xchg)
    T llvm_atomic_cmp_swap(T)(shared T* ptr, T cmp, T val, AtomicOrdering ordering = DefaultOrdering);


/// This intrinsic loads the value stored in memory at ptr and yields the value
/// from memory. It then stores the value in val in the memory at ptr.

pragma(LDC_atomic_rmw, "xchg")
    T llvm_atomic_swap(T)(shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// This intrinsic adds delta to the value stored in memory at ptr. It yields
/// the original value at ptr.

pragma(LDC_atomic_rmw, "add")
    T llvm_atomic_load_add(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// This intrinsic subtracts delta to the value stored in memory at ptr. It
/// yields the original value at ptr.

pragma(LDC_atomic_rmw, "sub")
    T llvm_atomic_load_sub(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// These intrinsics bitwise the operation (and, nand, or, xor) delta to the
/// value stored in memory at ptr. It yields the original value at ptr.

pragma(LDC_atomic_rmw, "and")
    T llvm_atomic_load_and(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "nand")
    T llvm_atomic_load_nand(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "or")
    T llvm_atomic_load_or(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "xor")
    T llvm_atomic_load_xor(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// These intrinsics takes the signed or unsigned minimum or maximum of delta
/// and the value stored in memory at ptr. It yields the original value at ptr.

pragma(LDC_atomic_rmw, "max")
    T llvm_atomic_load_max(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "min")
    T llvm_atomic_load_min(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "umax")
    T llvm_atomic_load_umax(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// ditto
pragma(LDC_atomic_rmw, "umin")
    T llvm_atomic_load_umin(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);


//
// ARITHMETIC-WITH-OVERFLOW INTRINSICS
//

///
struct OverflowRet(T) {
    static assert(is(T : long), T.stringof ~ " is not an integer type!");
    T result; ///
    bool overflow; ///
}

/// Signed and unsigned addition
pragma(LDC_intrinsic, "llvm.sadd.with.overflow.i#")
    OverflowRet!(T) llvm_sadd_with_overflow(T)(T lhs, T rhs);

pragma(LDC_intrinsic, "llvm.uadd.with.overflow.i#")
    OverflowRet!(T) llvm_uadd_with_overflow(T)(T lhs, T rhs); /// ditto


/// Signed and unsigned subtraction
pragma(LDC_intrinsic, "llvm.ssub.with.overflow.i#")
    OverflowRet!(T) llvm_ssub_with_overflow(T)(T lhs, T rhs);

pragma(LDC_intrinsic, "llvm.usub.with.overflow.i#")
    OverflowRet!(T) llvm_usub_with_overflow(T)(T lhs, T rhs); /// ditto


/// Signed and unsigned multiplication
pragma(LDC_intrinsic, "llvm.smul.with.overflow.i#")
    OverflowRet!(T) llvm_smul_with_overflow(T)(T lhs, T rhs);

pragma(LDC_intrinsic, "llvm.umul.with.overflow.i#")
    OverflowRet!(T) llvm_umul_with_overflow(T)(T lhs, T rhs);


//
// GENERAL INTRINSICS
//


/// This intrinsics is lowered to the target dependent trap instruction. If the
/// target does not have a trap instruction, this intrinsic will be lowered to
/// the call of the abort() function.

pragma(LDC_intrinsic, "llvm.trap")
    void llvm_trap();

version(INTRINSICS_FROM_302)
{
/// This intrinsic is lowered to code which is intended to cause an execution
/// trap with the intention of requesting the attention of a debugger.
pragma(LDC_intrinsic, "llvm.debugtrap")
    void llvm_debugtrap();
}

/// The llvm.expect intrinsic provides information about expected (the most
/// probable) value of val, which can be used by optimizers.
/// The llvm.expect intrinsic takes two arguments. The first argument is a
/// value. The second argument is an expected value, this needs to be a
/// constant value, variables are not allowed.

pragma(LDC_intrinsic, "llvm.expect.i#")
    T llvm_expect(T)(T val, T expected_val) if (__traits(isIntegral, T));
