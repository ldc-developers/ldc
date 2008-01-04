module llvm.intrinsic;

// code generator intrinsics
/*
pragma(LLVM_internal, "intrinsic", "llvm.returnaddress")
    void* llvm_returnaddress(uint level);
*/
pragma(LLVM_internal, "intrinsic", "llvm.frameaddress")
    void* llvm_frameaddress(uint level);
/*
pragma(LLVM_internal, "intrinsic", "llvm.stacksave")
    void* llvm_stacksave();

pragma(LLVM_internal, "intrinsic", "llvm.stackrestore")
    void llvm_stackrestore(void* ptr);

pragma(LLVM_internal, "intrinsic", "llvm.pcmarker")
    void llvm_pcmarker(uint id);

pragma(LLVM_internal, "intrinsic", "llvm.prefetch")
    void llvm_prefetch(void* ptr, uint rw, uint locality);
*/

pragma(LLVM_internal, "intrinsic", "llvm.readcyclecounter")
    ulong readcyclecounter();

// standard C intrinsics
pragma(LLVM_internal, "intrinsic", "llvm.memcpy.i32")
    void llvm_memcpy_i32(void* dst, void* src, uint len, uint alignment);

pragma(LLVM_internal, "intrinsic", "llvm.memcpy.i64")
    void llvm_memcpy_i64(void* dst, void* src, ulong len, uint alignment);

pragma(LLVM_internal, "intrinsic", "llvm.memmove.i32")
    void llvm_memmove_i32(void* dst, void* src, uint len, uint alignment);

pragma(LLVM_internal, "intrinsic", "llvm.memmove.i64")
    void llvm_memmove_i64(void* dst, void* src, ulong len, int alignment);

pragma(LLVM_internal, "intrinsic", "llvm.memset.i32")
    void llvm_memset_i32(void* dst, ubyte val, uint len, uint alignment);

pragma(LLVM_internal, "intrinsic", "llvm.memset.i64")
    void llvm_memset_i64(void* dst, ubyte val, ulong len, uint alignment);

pragma(LLVM_internal, "intrinsic", "llvm.sqrt.f32")
    float llvm_sqrt(float val);

pragma(LLVM_internal, "intrinsic", "llvm.sqrt.f64")
{
    double llvm_sqrt(double val);
    real llvm_sqrt(real val);
}

pragma(LLVM_internal, "intrinsic", "llvm.powi.f32")
    float llvm_powi(float val, int power);

pragma(LLVM_internal, "intrinsic", "llvm.powi.f64")
{
    double llvm_powi(double val, int power);
    real llvm_powi(real val, int power);
}

// bit manipulation intrinsics
pragma(LLVM_internal, "intrinsic", "llvm.bswap.i16.i16")
    ushort llvm_bswap(ushort val);

pragma(LLVM_internal, "intrinsic", "llvm.bswap.i32.i32")
    uint llvm_bswap(uint val);

pragma(LLVM_internal, "intrinsic", "llvm.bswap.i64.i64")
    ulong llvm_bswap(ulong val);

/*
pragma(LLVM_internal, "intrinsic", "llvm.ctpop.i8")
    uint llvm_ctpop_i8(ubyte src);

pragma(LLVM_internal, "intrinsic", "llvm.ctpop.i16")
    uint llvm_ctpop_i16(ushort src);

pragma(LLVM_internal, "intrinsic", "llvm.ctpop.i32")
    uint llvm_ctpop_i32(uint src);

pragma(LLVM_internal, "intrinsic", "llvm.ctpop.i64")
    uint llvm_ctpop_i64(ulong src);

pragma(LLVM_internal, "intrinsic", "llvm.ctlz.i8")
    uint llvm_ctlz_i8(ubyte src);

pragma(LLVM_internal, "intrinsic", "llvm.ctlz.i16")
    uint llvm_ctlz_i16(ushort src);

pragma(LLVM_internal, "intrinsic", "llvm.ctlz.i32")
    uint llvm_ctlz_i32(uint src);

pragma(LLVM_internal, "intrinsic", "llvm.ctlz.i64")
    uint llvm_ctlz_i64(ulong src);

pragma(LLVM_internal, "intrinsic", "llvm.cttz.i8")
    uint llvm_cttz_i8(ubyte src);

pragma(LLVM_internal, "intrinsic", "llvm.cttz.i16")
    uint llvm_cttz_i16(ushort src);

pragma(LLVM_internal, "intrinsic", "llvm.cttz.i32")
    uint llvm_cttz_i32(uint src);

pragma(LLVM_internal, "intrinsic", "llvm.cttz.i64")
    uint llvm_cttz_i64(ulong src);
*/

// atomic operations and synchronization intrinsics
// TODO
/*

//declare i8 @llvm.atomic.lcs.i8.i8p.i8.i8( i8* <ptr>, i8 <cmp>, i8 <val> )
pragma(LLVM_internal, "intrinsic", "llvm.atomic.lcs.i8.i8p.i8.i8")
    ubyte llvm_atomic_lcs_i8(void* ptr, ubyte cmp, ubyte val);

declare i16 @llvm.atomic.lcs.i16.i16p.i16.i16( i16* <ptr>, i16 <cmp>, i16 <val> )
declare i32 @llvm.atomic.lcs.i32.i32p.i32.i32( i32* <ptr>, i32 <cmp>, i32 <val> )
declare i64 @llvm.atomic.lcs.i64.i64p.i64.i64( i64* <ptr>, i64 <cmp>, i64 <val> )

declare i8 @llvm.atomic.ls.i8.i8p.i8( i8* <ptr>, i8 <val> )
declare i16 @llvm.atomic.ls.i16.i16p.i16( i16* <ptr>, i16 <val> )
declare i32 @llvm.atomic.ls.i32.i32p.i32( i32* <ptr>, i32 <val> )
declare i64 @llvm.atomic.ls.i64.i64p.i64( i64* <ptr>, i64 <val> )

declare i8 @llvm.atomic.las.i8.i8p.i8( i8* <ptr>, i8 <delta> )
declare i16 @llvm.atomic.las.i16.i16p.i16( i16* <ptr>, i16 <delta> )
declare i32 @llvm.atomic.las.i32.i32p.i32( i32* <ptr>, i32 <delta> )
declare i64 @llvm.atomic.las.i64.i64p.i64( i64* <ptr>, i64 <delta> )

declare i8 @llvm.atomic.lss.i8.i8.i8( i8* <ptr>, i8 <delta> )
declare i16 @llvm.atomic.lss.i16.i16.i16( i16* <ptr>, i16 <delta> )
declare i32 @llvm.atomic.lss.i32.i32.i32( i32* <ptr>, i32 <delta> )
declare i64 @llvm.atomic.lss.i64.i64.i64( i64* <ptr>, i64 <delta> )

declare void @llvm.memory.barrier( i1 <ll>, i1 <ls>, i1 <sl>, i1 <ss> )
*/
