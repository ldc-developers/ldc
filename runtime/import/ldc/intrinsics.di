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

version(X86)
    version = Reals_80Bit;
else version(X86_64)
    version = Reals_80Bit;

//
// CODE GENERATOR INTRINSICS
//


// The 'llvm.returnaddress' intrinsic attempts to compute a target-specific value indicating the return address of the current function or one of its callers. 

pragma(intrinsic, "llvm.returnaddress")
    void* llvm_returnaddress(uint level);


// The 'llvm.frameaddress' intrinsic attempts to return the target-specific frame pointer value for the specified stack frame. 

pragma(intrinsic, "llvm.frameaddress")
    void* llvm_frameaddress(uint level);


// The 'llvm.stacksave' intrinsic is used to remember the current state of the function stack, for use with llvm.stackrestore. This is useful for implementing language features like scoped automatic variable sized arrays in C99. 

pragma(intrinsic, "llvm.stacksave")
    void* llvm_stacksave();


// The 'llvm.stackrestore' intrinsic is used to restore the state of the function stack to the state it was in when the corresponding llvm.stacksave intrinsic executed. This is useful for implementing language features like scoped automatic variable sized arrays in C99. 

pragma(intrinsic, "llvm.stackrestore")
    void llvm_stackrestore(void* ptr);


// The 'llvm.prefetch' intrinsic is a hint to the code generator to insert a prefetch instruction if supported; otherwise, it is a noop. Prefetches have no effect on the behavior of the program but can change its performance characteristics.

pragma(intrinsic, "llvm.prefetch")
    void llvm_prefetch(void* ptr, uint rw, uint locality);


// The 'llvm.pcmarker' intrinsic is a method to export a Program Counter (PC) in a region of code to simulators and other tools. The method is target specific, but it is expected that the marker will use exported symbols to transmit the PC of the marker. The marker makes no guarantees that it will remain with any specific instruction after optimizations. It is possible that the presence of a marker will inhibit optimizations. The intended use is to be inserted after optimizations to allow correlations of simulation runs. 

pragma(intrinsic, "llvm.pcmarker")
    void llvm_pcmarker(uint id);


// The 'llvm.readcyclecounter' intrinsic provides access to the cycle counter register (or similar low latency, high accuracy clocks) on those targets that support it. On X86, it should map to RDTSC. On Alpha, it should map to RPCC. As the backing counters overflow quickly (on the order of 9 seconds on alpha), this should only be used for small timings. 

pragma(intrinsic, "llvm.readcyclecounter")
    ulong readcyclecounter();




//
// STANDARD C LIBRARY INTRINSICS
//


// The 'llvm.memcpy.*' intrinsics copy a block of memory from the source location to the destination location.
// Note that, unlike the standard libc function, the llvm.memcpy.* intrinsics do not return a value, and takes an extra alignment argument.

pragma(intrinsic, "llvm.memcpy.i32")
    void llvm_memcpy_i32(void* dst, void* src, uint len, uint alignment);
pragma(intrinsic, "llvm.memcpy.i64")
    void llvm_memcpy_i64(void* dst, void* src, ulong len, uint alignment);


// The 'llvm.memmove.*' intrinsics move a block of memory from the source location to the destination location. It is similar to the 'llvm.memcpy' intrinsic but allows the two memory locations to overlap.
// Note that, unlike the standard libc function, the llvm.memmove.* intrinsics do not return a value, and takes an extra alignment argument.

pragma(intrinsic, "llvm.memmove.i32")
    void llvm_memmove_i32(void* dst, void* src, uint len, uint alignment);
pragma(intrinsic, "llvm.memmove.i64")
    void llvm_memmove_i64(void* dst, void* src, ulong len, int alignment);


// The 'llvm.memset.*' intrinsics fill a block of memory with a particular byte value.
// Note that, unlike the standard libc function, the llvm.memset intrinsic does not return a value, and takes an extra alignment argument.

pragma(intrinsic, "llvm.memset.i32")
    void llvm_memset_i32(void* dst, ubyte val, uint len, uint alignment);
pragma(intrinsic, "llvm.memset.i64")
    void llvm_memset_i64(void* dst, ubyte val, ulong len, uint alignment);


// The 'llvm.sqrt' intrinsics return the sqrt of the specified operand, returning the same value as the libm 'sqrt' functions would. Unlike sqrt in libm, however, llvm.sqrt has undefined behavior for negative numbers other than -0.0 (which allows for better optimization, because there is no need to worry about errno being set). llvm.sqrt(-0.0) is defined to return -0.0 like IEEE sqrt. 

pragma(intrinsic, "llvm.sqrt.f32")
    float llvm_sqrt_f32(float val);
pragma(intrinsic, "llvm.sqrt.f64")
    double llvm_sqrt_f64(double val);

version(Reals_80Bit)
{
    pragma(intrinsic, "llvm.sqrt.f80")
        real llvm_sqrt_f80(real val);
    alias llvm_sqrt_f80 llvm_sqrt_real;
}
else
{
    pragma(intrinsic, "llvm.sqrt.f64")
        real llvm_sqrt_real(real val);
}


// The 'llvm.sin.*' intrinsics return the sine of the operand. 

pragma(intrinsic, "llvm.sin.f32")
    float llvm_sin_f32(float val);
pragma(intrinsic, "llvm.sin.f64")
    double llvm_sin_f64(double val);

version(Reals_80Bit)
{
    pragma(intrinsic, "llvm.sin.f80")
        real llvm_sin_f80(real val);
    alias llvm_sin_f80 llvm_sin_real;
}
else
{
    pragma(intrinsic, "llvm.sin.f64")
        real llvm_sin_real(real val);
}


// The 'llvm.cos.*' intrinsics return the cosine of the operand. 

pragma(intrinsic, "llvm.cos.f32")
    float llvm_cos_f32(float val);
pragma(intrinsic, "llvm.cos.f64")
    double llvm_cos_f64(double val);

version(Reals_80Bit)
{
    pragma(intrinsic, "llvm.cos.f80")
        real llvm_cos_f80(real val);
    alias llvm_cos_f80 llvm_cos_real;
}
else
{
    pragma(intrinsic, "llvm.cos.f64")
        real llvm_cos_real(real val);
}


// The 'llvm.powi.*' intrinsics return the first operand raised to the specified (positive or negative) power. The order of evaluation of multiplications is not defined. When a vector of floating point type is used, the second argument remains a scalar integer value. 

pragma(intrinsic, "llvm.powi.f32")
    float llvm_powi_f32(float val, int power);

pragma(intrinsic, "llvm.powi.f64")
    double llvm_powi_f64(double val, int power);

version(Reals_80Bit)
{
    pragma(intrinsic, "llvm.powi.f80")
        real llvm_powi_f80(real val, int power);
    alias llvm_powi_f80 llvm_powi_real;
}
else
{
    pragma(intrinsic, "llvm.powi.f64")
        real llvm_powi_real(real val, int power);
}


// The 'llvm.pow.*' intrinsics return the first operand raised to the specified (positive or negative) power. 

pragma(intrinsic, "llvm.pow.f32")
    float llvm_pow_f32(float val, float power);

pragma(intrinsic, "llvm.pow.f64")
    double llvm_pow_f64(double val, double power);

version(Reals_80Bit)
{
    pragma(intrinsic, "llvm.pow.f80")
        real llvm_pow_f80(real val, real power);
    alias llvm_pow_f80 llvm_pow_real;
}
else
{
    pragma(intrinsic, "llvm.pow.f64")
        real llvm_pow_real(real val, real power);
}


//
// BIT MANIPULATION INTRINSICS
//

// The 'llvm.bswap' family of intrinsics is used to byte swap integer values with an even number of bytes (positive multiple of 16 bits). These are useful for performing operations on data that is not in the target's native byte order. 

pragma(intrinsic, "llvm.bswap.i16.i16")
    ushort llvm_bswap_i16(ushort val);

pragma(intrinsic, "llvm.bswap.i32.i32")
    uint llvm_bswap_i32(uint val);

pragma(intrinsic, "llvm.bswap.i64.i64")
    ulong llvm_bswap_i64(ulong val);


// The 'llvm.ctpop' family of intrinsics counts the number of bits set in a value. 

pragma(intrinsic, "llvm.ctpop.i8")
    ubyte llvm_ctpop_i8(ubyte src);

pragma(intrinsic, "llvm.ctpop.i16")
    ushort llvm_ctpop_i16(ushort src);

pragma(intrinsic, "llvm.ctpop.i32")
    uint llvm_ctpop_i32(uint src);

pragma(intrinsic, "llvm.ctpop.i64")
    ulong llvm_ctpop_i64(ulong src);


// The 'llvm.ctlz' family of intrinsic functions counts the number of leading zeros in a variable. 

pragma(intrinsic, "llvm.ctlz.i8")
    ubyte llvm_ctlz_i8(ubyte src);

pragma(intrinsic, "llvm.ctlz.i16")
    ushort llvm_ctlz_i16(ushort src);

pragma(intrinsic, "llvm.ctlz.i32")
    uint llvm_ctlz_i32(uint src);

pragma(intrinsic, "llvm.ctlz.i64")
    ulong llvm_ctlz_i64(ulong src);


// The 'llvm.cttz' family of intrinsic functions counts the number of trailing zeros. 

pragma(intrinsic, "llvm.cttz.i8")
    ubyte llvm_cttz_i8(ubyte src);

pragma(intrinsic, "llvm.cttz.i16")
    ushort llvm_cttz_i16(ushort src);

pragma(intrinsic, "llvm.cttz.i32")
    uint llvm_cttz_i32(uint src);

pragma(intrinsic, "llvm.cttz.i64")
    ulong llvm_cttz_i64(ulong src);


// The 'llvm.part.select' family of intrinsic functions selects a range of bits from an integer value and returns them in the same bit width as the original value.

pragma(intrinsic, "llvm.part.select.i8")
    ubyte llvm_part_select_i(ubyte val, uint loBit, uint hiBit);

pragma(intrinsic, "llvm.part.select.i16")
    ushort llvm_part_select_i(ushort val, uint loBit, uint hiBit);

pragma(intrinsic, "llvm.part.select.i32")
    uint llvm_part_select_i(uint val, uint loBit, uint hiBit);

pragma(intrinsic, "llvm.part.select.i64")
    ulong llvm_part_select_i(ulong val, uint loBit, uint hiBit);


// The 'llvm.part.set' family of intrinsic functions replaces a range of bits in an integer value with another integer value. It returns the integer with the replaced bits.

// TODO
// declare i17 @llvm.part.set.i17.i9 (i17 %val, i9 %repl, i32 %lo, i32 %hi)
// declare i29 @llvm.part.set.i29.i9 (i29 %val, i9 %repl, i32 %lo, i32 %hi)




//
// ATOMIC OPERATIONS AND SYNCHRONIZATION INTRINSICS
//

// The llvm.memory.barrier intrinsic guarantees ordering between specific pairs of memory access types.

pragma(intrinsic, "llvm.memory.barrier")
    void llvm_memory_barrier(bool ll, bool ls, bool sl, bool ss, bool device);

// This loads a value in memory and compares it to a given value. If they are equal, it stores a new value into the memory.

pragma(intrinsic, "llvm.atomic.cmp.swap.i#.p0i#")
    T llvm_atomic_cmp_swap(T)(T* ptr, T cmp, T val);

// This intrinsic loads the value stored in memory at ptr and yields the value from memory. It then stores the value in val in the memory at ptr.

pragma(intrinsic, "llvm.atomic.swap.i#.p0i#")
    T llvm_atomic_swap(T)(T* ptr, T val);

// This intrinsic adds delta to the value stored in memory at ptr. It yields the original value at ptr.

pragma(intrinsic, "llvm.atomic.load.add.i#.p0i#")
    T llvm_atomic_load_add(T)(T* ptr, T val);

// This intrinsic subtracts delta to the value stored in memory at ptr. It yields the original value at ptr.

pragma(intrinsic, "llvm.atomic.load.sub.i#.p0i#")
    T llvm_atomic_load_sub(T)(T* ptr, T val);

// These intrinsics bitwise the operation (and, nand, or, xor) delta to the value stored in memory at ptr. It yields the original value at ptr.

pragma(intrinsic, "llvm.atomic.load.and.i#.p0i#")
    T llvm_atomic_load_and(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.nand.i#.p0i#")
    T llvm_atomic_load_nand(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.or.i#.p0i#")
    T llvm_atomic_load_or(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.xor.i#.p0i#")
    T llvm_atomic_load_xor(T)(T* ptr, T val);

// These intrinsics takes the signed or unsigned minimum or maximum of delta and the value stored in memory at ptr. It yields the original value at ptr. 

pragma(intrinsic, "llvm.atomic.load.max.i#.p0i#")
    T llvm_atomic_load_max(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.min.i#.p0i#")
    T llvm_atomic_load_min(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.umax.i#.p0i#")
    T llvm_atomic_load_umax(T)(T* ptr, T val);
pragma(intrinsic, "llvm.atomic.load.umin.i#.p0i#")
    T llvm_atomic_load_umin(T)(T* ptr, T val);


//
// ARITHMETIC-WITH-OVERFLOW INTRINSICS
//

struct OverflowRet(T) {
    static assert(is(T : int), T.stringof ~ " is not an integer type!");
    T result;
    bool overflow;
}

// Signed and unsigned addition
pragma(intrinsic, "llvm.sadd.with.overflow.i#")
    OverflowRet!(T) llvm_sadd_with_overflow(T)(T lhs, T rhs);

pragma(intrinsic, "llvm.uadd.with.overflow.i#")
    OverflowRet!(T) llvm_uadd_with_overflow(T)(T lhs, T rhs);

// Signed and unsigned subtraction
pragma(intrinsic, "llvm.ssub.with.overflow.i#")
    OverflowRet!(T) llvm_ssub_with_overflow(T)(T lhs, T rhs);

pragma(intrinsic, "llvm.usub.with.overflow.i#")
    OverflowRet!(T) llvm_usub_with_overflow(T)(T lhs, T rhs);

// Signed and unsigned multiplication
pragma(intrinsic, "llvm.smul.with.overflow.i#")
    OverflowRet!(T) llvm_smul_with_overflow(T)(T lhs, T rhs);

/* Note: LLVM documentations says:
 *  Warning: 'llvm.umul.with.overflow' is badly broken.
 *  It is actively being fixed, but it should not currently be used!
 *
 * See: http://llvm.org/docs/LangRef.html#int_umul_overflow
 */
pragma(intrinsic, "llvm.umul.with.overflow.i#")
    OverflowRet!(T) llvm_umul_with_overflow(T)(T lhs, T rhs);


//
// GENERAL INTRINSICS
//


// This intrinsics is lowered to the target dependent trap instruction. If the target does not have a trap instruction, this intrinsic will be lowered to the call of the abort() function. 

pragma(intrinsic, "llvm.trap")
    void llvm_trap();
