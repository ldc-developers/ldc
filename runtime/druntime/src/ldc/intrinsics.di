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
version (LDC)
{
    // OK
}
else
{
    static assert(false, "This module is only valid for LDC");
}

     version (LDC_LLVM_1100) enum LLVM_version = 1100;
else version (LDC_LLVM_1101) enum LLVM_version = 1101;
else version (LDC_LLVM_1200) enum LLVM_version = 1200;
else version (LDC_LLVM_1300) enum LLVM_version = 1300;
else version (LDC_LLVM_1400) enum LLVM_version = 1400;
else version (LDC_LLVM_1500) enum LLVM_version = 1500;
else version (LDC_LLVM_1600) enum LLVM_version = 1600;
else version (LDC_LLVM_1700) enum LLVM_version = 1700;
else version (LDC_LLVM_1800) enum LLVM_version = 1800;
else static assert(false, "LDC LLVM version not supported");

enum LLVM_atleast(int major) = (LLVM_version >= major * 100);

// All intrinsics are nothrow and @nogc. The codegen intrinsics are not categorized
// any further (they probably could), the rest is pure (aborting is fine by
// definition; memcpy and friends can be viewed as weakly pure, just as e.g.
// strlen() is marked weakly pure as well) and mostly @safe.
nothrow:
@nogc:

version(LDC_LLVM_OpaquePointers)
    private enum p0i8 = "p0";
else
    private enum p0i8 = "p0i8";

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
pragma(LDC_intrinsic, "llvm.frameaddress."~p0i8)
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
pragma(LDC_intrinsic, "llvm.prefetch."~p0i8)
    void llvm_prefetch(const(void)* ptr, uint rw, uint locality, uint cachetype) pure @safe;

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
    ulong llvm_readcyclecounter() @safe;

// Backwards compatibility - but it is doubtful whether somebody actually ever
// used that intrinsic.
alias llvm_readcyclecounter readcyclecounter;

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

/// The ‘llvm.thread.pointer‘ intrinsic returns a pointer to the TLS area for the
/// current thread. The exact semantics of this value are target specific: it may
/// point to the start of TLS area, to the end, or somewhere in the middle. Depending
/// on the target, this intrinsic may read a register, call a helper function, read
/// from an alternate memory space, or perform other operations necessary to locate
/// the TLS area. Not all targets support this intrinsic.
pragma(LDC_intrinsic, "llvm.thread.pointer")
    void* llvm_thread_pointer();

//
// STANDARD C LIBRARY INTRINSICS
//

pure:

// The alignment parameter was removed from these memory intrinsics in LLVM 7.0. Instead, alignment
// can be specified as an attribute on the ptr arguments.

/// The 'llvm.memcpy.*' intrinsics copy a block of memory from the source
/// location to the destination location.
/// Note that, unlike the standard libc function, the llvm.memcpy.* intrinsics do
/// not return a value.
pragma(LDC_intrinsic, "llvm.memcpy."~p0i8~"."~p0i8~".i#")
    void llvm_memcpy(T)(void* dst, const(void)* src, T len, bool volatile_ = false)
        if (__traits(isIntegral, T));


/// The 'llvm.memmove.*' intrinsics move a block of memory from the source
/// location to the destination location. It is similar to the 'llvm.memcpy'
/// intrinsic but allows the two memory locations to overlap.
/// Note that, unlike the standard libc function, the llvm.memmove.* intrinsics
/// do not return a value.
pragma(LDC_intrinsic, "llvm.memmove."~p0i8~"."~p0i8~".i#")
    void llvm_memmove(T)(void* dst, const(void)* src, T len, bool volatile_ = false)
        if (__traits(isIntegral, T));

/// The 'llvm.memset.*' intrinsics fill a block of memory with a particular byte
/// value.
/// Note that, unlike the standard libc function, the llvm.memset intrinsic does
/// not return a value.
pragma(LDC_intrinsic, "llvm.memset."~p0i8~".i#")
    void llvm_memset(T)(void* dst, ubyte val, T len, bool volatile_ = false)
        if (__traits(isIntegral, T));

/// Convenience function that discards the alignment parameter and calls the 'llvm.memcpy.*' intrinsic.
/// This function is here to support the function signature of the pre-LLVM7.0 intrinsic.
pragma(inline, true)
void llvm_memcpy(T)(void* dst, const(void)* src, T len, uint alignment, bool volatile_ = false)
    if (__traits(isIntegral, T))
{
    if (volatile_)
        llvm_memcpy!T(dst, src, len, true);
    else
        llvm_memcpy!T(dst, src, len, false);
}
/// Convenience function that discards the alignment parameter and calls the 'llvm.memmove.*' intrinsic.
/// This function is here to support the function signature of the pre-LLVM7.0 intrinsic.
pragma(inline, true)
void llvm_memmove(T)(void* dst, const(void)* src, T len, uint alignment, bool volatile_ = false)
    if (__traits(isIntegral, T))
{
    if (volatile_)
        llvm_memmove!T(dst, src, len, true);
    else
        llvm_memmove!T(dst, src, len, false);
}
/// Convenience function that discards the alignment parameter and calls the 'llvm.memset.*' intrinsic.
/// This function is here to support the function signature of the pre-LLVM7.0 intrinsic.
pragma(inline, true)
void llvm_memset(T)(void* dst, ubyte val, T len, uint alignment, bool volatile_ = false)
    if (__traits(isIntegral, T))
{
    if (volatile_)
        llvm_memset!T(dst, val, len, true);
    else
        llvm_memset!T(dst, val, len, false);
}

@safe:

/// The 'llvm.sqrt' intrinsics return the sqrt of the specified operand,
/// returning the same value as the libm 'sqrt' functions would. Unlike sqrt in
/// libm, however, llvm.sqrt has undefined behavior for negative numbers other
/// than -0.0 (which allows for better optimization, because there is no need to
/// worry about errno being set). llvm.sqrt(-0.0) is defined to return -0.0 like
/// IEEE sqrt.
pragma(LDC_intrinsic, "llvm.sqrt.f#")
    T llvm_sqrt(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.sin.*' intrinsics return the sine of the operand.
pragma(LDC_intrinsic, "llvm.sin.f#")
    T llvm_sin(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.cos.*' intrinsics return the cosine of the operand.
pragma(LDC_intrinsic, "llvm.cos.f#")
    T llvm_cos(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.powi.*' intrinsics return the first operand raised to the specified
/// (positive or negative) power. The order of evaluation of multiplications is
/// not defined. When a vector of floating point type is used, the second
/// argument remains a scalar integer value.
pragma(LDC_intrinsic, "llvm.powi.f#")
    T llvm_powi(T)(T val, int power)
        if (__traits(isFloating, T));

/// The 'llvm.pow.*' intrinsics return the first operand raised to the specified
/// (positive or negative) power.
pragma(LDC_intrinsic, "llvm.pow.f#")
    T llvm_pow(T)(T val, T power)
        if (__traits(isFloating, T));

/// The 'llvm.exp.*' intrinsics perform the exp function.
pragma(LDC_intrinsic, "llvm.exp.f#")
    T llvm_exp(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.log.*' intrinsics perform the log function.
pragma(LDC_intrinsic, "llvm.log.f#")
    T llvm_log(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.fma.*' intrinsics perform the fused multiply-add operation.
pragma(LDC_intrinsic, "llvm.fma.f#")
    T llvm_fma(T)(T vala, T valb, T valc)
        if (__traits(isFloating, T));

/// The 'llvm.fabs.*' intrinsics return the absolute value of the operand.
pragma(LDC_intrinsic, "llvm.fabs.f#")
    T llvm_fabs(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.floor.*' intrinsics return the floor of the operand.
pragma(LDC_intrinsic, "llvm.floor.f#")
    T llvm_floor(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.exp2.*' intrinsics perform the exp2 function.
pragma(LDC_intrinsic, "llvm.exp2.f#")
    T llvm_exp2(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.log10.*' intrinsics perform the log10 function.
pragma(LDC_intrinsic, "llvm.log10.f#")
    T llvm_log10(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.log2.*' intrinsics perform the log2 function.
pragma(LDC_intrinsic, "llvm.log2.f#")
    T llvm_log2(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.ceil.*' intrinsics return the ceiling of the operand.
pragma(LDC_intrinsic, "llvm.ceil.f#")
    T llvm_ceil(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.trunc.*' intrinsics returns the operand rounded to the nearest integer not larger in magnitude than the operand.
pragma(LDC_intrinsic, "llvm.trunc.f#")
    T llvm_trunc(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.rint.*' intrinsics returns the operand rounded to the nearest integer. It may raise an inexact floating-point exception if the operand isn't an integer.
pragma(LDC_intrinsic, "llvm.rint.f#")
    T llvm_rint(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.nearbyint.*' intrinsics returns the operand rounded to the nearest integer.
pragma(LDC_intrinsic, "llvm.nearbyint.f#")
    T llvm_nearbyint(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.copysign.*' intrinsics return a value with the magnitude of the first operand and the sign of the second operand.
pragma(LDC_intrinsic, "llvm.copysign.f#")
    T llvm_copysign(T)(T mag, T sgn)
        if (__traits(isFloating, T));

/// The 'llvm.round.*' intrinsics returns the operand rounded to the nearest integer.
pragma(LDC_intrinsic, "llvm.round.f#")
    T llvm_round(T)(T val)
        if (__traits(isFloating, T));

/// The 'llvm.fmuladd.*' intrinsic functions represent multiply-add expressions
/// that can be fused if the code generator determines that the fused expression
///  would be legal and efficient.
pragma(LDC_intrinsic, "llvm.fmuladd.f#")
    T llvm_fmuladd(T)(T vala, T valb, T valc)
        if (__traits(isFloating, T));

/// The ‘llvm.minnum.*‘ intrinsics return the minimum of the two arguments.
/// Follows the IEEE-754 semantics for minNum, which also match for libm’s fmin.
/// If either operand is a NaN, returns the other non-NaN operand. Returns NaN
/// only if both operands are NaN. If the operands compare equal, returns a value
/// that compares equal to both operands. This means that fmin(+/-0.0, +/-0.0)
/// could return either -0.0 or 0.0.
pragma(LDC_intrinsic, "llvm.minnum.f#")
    T llvm_minnum(T)(T vala, T valb)
        if (__traits(isFloating, T));

/// The ‘llvm.maxnum.*‘ intrinsics return the maximum of the two arguments.
/// Follows the IEEE-754 semantics for maxNum, which also match for libm’s fmax.
/// If either operand is a NaN, returns the other non-NaN operand. Returns NaN
/// only if both operands are NaN. If the operands compare equal, returns a value
/// that compares equal to both operands. This means that fmax(+/-0.0, +/-0.0)
/// could return either -0.0 or 0.0.
pragma(LDC_intrinsic, "llvm.maxnum.f#")
    T llvm_maxnum(T)(T vala, T valb)
        if (__traits(isFloating, T));

/// The ‘llvm.minimum.*’ intrinsics return the minimum of the two arguments, propagating
/// NaNs and treating -0.0 as less than +0.0.
/// If either operand is a NaN, returns NaN. Otherwise returns the lesser of the two arguments.
/// -0.0 is considered to be less than +0.0 for this intrinsic. These are the
/// semantics specified in the draft of IEEE 754-2018.
pragma(LDC_intrinsic, "llvm.minimum.f#")
    T llvm_minimum(T)(T vala, T valb)
        if (__traits(isFloating, T));

/// The ‘llvm.maximum.*’ intrinsics return the maximum of the two arguments, propagating
/// NaNs and treating -0.0 as less than +0.0.
/// If either operand is a NaN, returns NaN. Otherwise returns the greater of the two arguments.
/// -0.0 is considered to be less than +0.0 for this intrinsic. Note that these are the
/// semantics specified in the draft of IEEE 754-2018.
pragma(LDC_intrinsic, "llvm.maximum.f#")
    T llvm_maximum(T)(T vala, T valb)
        if (__traits(isFloating, T));

//
// BIT MANIPULATION INTRINSICS
//

/// The 'llvm.bitreverse' family of intrinsics is used to reverse the bitpattern
/// of an integer value; for example 0b10110110 becomes 0b01101101.
pragma(LDC_intrinsic, "llvm.bitreverse.i#")
    T llvm_bitreverse(T)(T val)
        if (__traits(isIntegral, T));

/// The 'llvm.bswap' family of intrinsics is used to byte swap integer values
/// with an even number of bytes (positive multiple of 16 bits). These are
/// useful for performing operations on data that is not in the target's native
/// byte order.
pragma(LDC_intrinsic, "llvm.bswap.i#")
    T llvm_bswap(T)(T val)
        if (__traits(isIntegral, T) && T.sizeof >= 2);

/// The 'llvm.ctpop' family of intrinsics counts the number of bits set in a
/// value.
pragma(LDC_intrinsic, "llvm.ctpop.i#")
    T llvm_ctpop(T)(T src)
        if (__traits(isIntegral, T));

/// The 'llvm.ctlz' family of intrinsic functions counts the number of leading
/// zeros in a variable.
pragma(LDC_intrinsic, "llvm.ctlz.i#")
    T llvm_ctlz(T)(T src, bool isZeroUndefined)
        if (__traits(isIntegral, T));

/// The 'llvm.cttz' family of intrinsic functions counts the number of trailing
/// zeros.
pragma(LDC_intrinsic, "llvm.cttz.i#")
    T llvm_cttz(T)(T src, bool isZeroUndefined)
        if (__traits(isIntegral, T));

/// The ‘llvm.fshl’ family of intrinsic functions performs a funnel shift left:
/// the first two values are concatenated as `{ a : b }` (`a` is the most
/// significant bits of the wide value), the combined value is shifted left,
/// and the most significant bits are extracted to produce a result that is the
/// same size as the original arguments. If the first 2 arguments are identical,
/// this is equivalent to a rotate left operation. For vector types, the
/// operation occurs for each element of the vector. The shift argument is
/// treated as an unsigned amount modulo the element size of the arguments.
pragma(LDC_intrinsic, "llvm.fshl.i#")
    T llvm_fshl(T)(T a, T b, T shift)
        if (__traits(isIntegral, T));

/// The ‘llvm.fshr’ family of intrinsic functions performs a funnel shift right:
/// the first two values are concatenated as `{ a : b }` (`a` is the most
/// significant bits of the wide value), the combined value is shifted right,
/// and the least significant bits are extracted to produce a result that is the
/// same size as the original arguments. If the first 2 arguments are identical,
/// this is equivalent to a rotate right operation. For vector types, the
/// operation occurs for each element of the vector. The shift argument is
/// treated as an unsigned amount modulo the element size of the arguments.
pragma(LDC_intrinsic, "llvm.fshr.i#")
    T llvm_fshr(T)(T a, T b, T shift)
        if (__traits(isIntegral, T));


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
}
alias DefaultOrdering = AtomicOrdering.SequentiallyConsistent;

enum SynchronizationScope {
  SingleThread = 0,
  CrossThread  = 1,
  Default = CrossThread
}

enum AtomicRmwSizeLimit = size_t.sizeof;

/// Used to introduce happens-before edges between operations.
pragma(LDC_fence)
    void llvm_memory_fence(AtomicOrdering ordering = DefaultOrdering,
                           SynchronizationScope syncScope = SynchronizationScope.Default);

/// Atomically loads and returns a value from memory at ptr.
pragma(LDC_atomic_load)
    T llvm_atomic_load(T)(in shared T* ptr, AtomicOrdering ordering = DefaultOrdering);

/// Atomically stores val in memory at ptr.
pragma(LDC_atomic_store)
    void llvm_atomic_store(T)(T val, shared T* ptr, AtomicOrdering ordering = DefaultOrdering);

///
struct CmpXchgResult(T) {
    T previousValue; ///
    bool exchanged; ///
}

/// Loads a value from memory at ptr and compares it to cmp.
/// If they are equal, it stores val in memory at ptr.
/// This is all performed as single atomic operation.
pragma(LDC_atomic_cmp_xchg)
    CmpXchgResult!T llvm_atomic_cmp_xchg(T)(
        shared T* ptr, T cmp, T val,
        AtomicOrdering successOrdering = DefaultOrdering,
        AtomicOrdering failureOrdering = DefaultOrdering,
        bool weak = false);

/// Atomically sets *ptr = val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "xchg")
    T llvm_atomic_rmw_xchg(T)(shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr += val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "add")
    T llvm_atomic_rmw_add(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr -= val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "sub")
    T llvm_atomic_rmw_sub(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr &= val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "and")
    T llvm_atomic_rmw_and(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr = ~(*ptr & val) and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "nand")
    T llvm_atomic_rmw_nand(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr |= val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "or")
    T llvm_atomic_rmw_or(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr ^= val and returns the previous *ptr value.
pragma(LDC_atomic_rmw, "xor")
    T llvm_atomic_rmw_xor(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr = (*ptr > val ? *ptr : val) using a signed comparison.
/// Returns the previous *ptr value.
pragma(LDC_atomic_rmw, "max")
    T llvm_atomic_rmw_max(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr = (*ptr < val ? *ptr : val) using a signed comparison.
/// Returns the previous *ptr value.
pragma(LDC_atomic_rmw, "min")
    T llvm_atomic_rmw_min(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr = (*ptr > val ? *ptr : val) using an unsigned comparison.
/// Returns the previous *ptr value.
pragma(LDC_atomic_rmw, "umax")
    T llvm_atomic_rmw_umax(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);

/// Atomically sets *ptr = (*ptr < val ? *ptr : val) using an unsigned comparison.
/// Returns the previous *ptr value.
pragma(LDC_atomic_rmw, "umin")
    T llvm_atomic_rmw_umin(T)(in shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);



//
// ARITHMETIC-WITH-OVERFLOW INTRINSICS
//

///
struct OverflowRet(T) {
    static assert((is(ucent) && is(T : cent)) || is(T : long), T.stringof ~ " is not an integer type!");
    T result; ///
    bool overflow; ///
}

/// Signed addition
pragma(LDC_intrinsic, "llvm.sadd.with.overflow.i#")
    OverflowRet!(T) llvm_sadd_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Unsigned addition
pragma(LDC_intrinsic, "llvm.uadd.with.overflow.i#")
    OverflowRet!(T) llvm_uadd_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Signed subtraction
pragma(LDC_intrinsic, "llvm.ssub.with.overflow.i#")
    OverflowRet!(T) llvm_ssub_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Unsigned subtraction
pragma(LDC_intrinsic, "llvm.usub.with.overflow.i#")
    OverflowRet!(T) llvm_usub_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Signed multiplication
pragma(LDC_intrinsic, "llvm.smul.with.overflow.i#")
    OverflowRet!(T) llvm_smul_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Unsigned multiplication
pragma(LDC_intrinsic, "llvm.umul.with.overflow.i#")
    OverflowRet!(T) llvm_umul_with_overflow(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

//
// SATURATION ARITHMETIC INTRINSICS
//
// Saturation arithmetic is a version of arithmetic in which operations are
// limited to a fixed range between a minimum and maximum value. If the result of
// an operation is greater than the maximum value, the result is set (or
// "clamped") to this maximum. If it is below the minimum, it is clamped to this
// minimum.
//

/// Signed Saturation Addition
/// The maximum value this operation can clamp to is the largest signed value
/// representable by the bit width of the arguments. The minimum value is the
/// smallest signed value representable by this bit width.
pragma(LDC_intrinsic, "llvm.sadd.sat.i#")
    T llvm_sadd_sat(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Unsigned Saturation Addition
/// The maximum value this operation can clamp to is the largest unsigned value
/// representable by the bit width of the arguments. Because this is an unsigned
/// operation, the result will never saturate towards zero.
pragma(LDC_intrinsic, "llvm.uadd.sat.i#")
    T llvm_uadd_sat(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Signed Saturation Subtraction
/// The maximum value this operation can clamp to is the largest signed value
/// representable by the bit width of the arguments. The minimum value is the
/// smallest signed value representable by this bit width.
pragma(LDC_intrinsic, "llvm.ssub.sat.i#")
    T llvm_ssub_sat(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

/// Unsigned Saturation Subtraction
/// The minimum value this operation can clamp to is 0, which is the smallest
/// unsigned value representable by the bit width of the unsigned arguments.
/// Because this is an unsigned operation, the result will never saturate towards
/// the largest possible value representable by this bit width.
pragma(LDC_intrinsic, "llvm.usub.sat.i#")
    T llvm_usub_sat(T)(T lhs, T rhs)
        if (__traits(isIntegral, T));

//
// GENERAL INTRINSICS
//

/// This intrinsics is lowered to the target dependent trap instruction. If the
/// target does not have a trap instruction, this intrinsic will be lowered to
/// the call of the abort() function.
pragma(LDC_intrinsic, "llvm.trap")
    void llvm_trap();

/// This intrinsic is lowered to code which is intended to cause an execution
/// trap with the intention of requesting the attention of a debugger.
pragma(LDC_intrinsic, "llvm.debugtrap")
    void llvm_debugtrap();

/// Provides information about the expected (that is, most probable) runtime
/// value of an integer expression to the optimizer.
///
/// Params:
///     val = The runtime value, of integer type.
///     expectedVal = The expected value of `val` – needs to be a constant!
pragma(LDC_intrinsic, "llvm.expect.i#")
    T llvm_expect(T)(T val, T expectedVal)
        if (__traits(isIntegral, T));

/// LLVM optimizer treats this intrinsic as having side effect, so it can be
/// inserted into a loop to indicate that the loop shouldn't be assumed to
/// terminate even if it's an infinite loop with no other side effect.
pragma(LDC_intrinsic, "llvm.sideeffect")
    void llvm_sideeffect();

version (WebAssembly)
{
/// Grows memory by a given delta and returns the previous size, or -1 if enough
/// memory cannot be allocated.
///
/// Note:
///     In the current version of WebAssembly, all memory instructions implicitly
///     operate on memory index 0. This restriction may be lifted in future versions.
///
/// https://webassembly.github.io/spec/core/exec/instructions.html#exec-memory-grow
pragma(LDC_intrinsic, "llvm.wasm.memory.grow.i32")
    int llvm_wasm_memory_grow(int mem, int delta);

/// Returns the current size of memory.
///
/// Note:
///     In the current version of WebAssembly, all memory instructions implicitly
///     operate on memory index 0. This restriction may be lifted in future versions.
///
/// https://webassembly.github.io/spec/core/exec/instructions.html#exec-memory-size
pragma(LDC_intrinsic, "llvm.wasm.memory.size.i32")
    int llvm_wasm_memory_size(int mem);
} // version (WebAssembly)
