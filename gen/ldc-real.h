//===-- gen/ldc-real.h - Interface of real_t for LDC ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Implements a longdouble type for LDC.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LDC_REAL_H
#define LDC_GEN_LDC_REAL_H

#undef min
#undef max

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

namespace ldc
{

class longdouble
{
public:
    // List of supported floating point semantics in D.
    enum FloatSemanticsInD
    {
        Float,
        Double,
        LongDouble,
        NumModes
    };

private:
    // Space for a llvm:APFloat object.
    // This class must not have a default constructor. Therefore we have to
    // manage the payload on our own.
    unsigned char mem[sizeof(llvm::APFloat)];

    llvm::APFloat *value()
    {
        return reinterpret_cast<llvm::APFloat *>(&this->mem);
    }

    const llvm::APFloat *value() const
    {
        return reinterpret_cast<const llvm::APFloat *>(&this->mem);
    }

    longdouble &init(const llvm::APFloat &other)
    {
        new (&this->mem) llvm::APFloat(other);
        return *this;
    }

    longdouble &init(const longdouble &other)
    {
        return init(*other.value());
    }

public:
    static const llvm::fltSemantics &getFltSemantics();

    template<typename T>
    longdouble& operator=(T x)
    {
        set(x);
        return *this;
    }

    longdouble operator+(const longdouble &r)
    {
        longdouble tmp;
        tmp.init(*this).value()->add(*r.value(), llvm::APFloat::rmNearestTiesToEven);
        return tmp;
    }

    longdouble operator-(const longdouble &r)
    {
        longdouble tmp;
        tmp.init(*this).value()->subtract(*r.value(), llvm::APFloat::rmNearestTiesToEven);
        return tmp;
    }

    longdouble operator-()
    {
        longdouble tmp;
        tmp.init(*this).value()->changeSign();
        return tmp;
    }

    longdouble operator*(const longdouble &r)
    {
        longdouble tmp;
        tmp.init(*this).value()->multiply(*r.value(), llvm::APFloat::rmNearestTiesToEven);
        return tmp;
    }

    longdouble operator/(const longdouble &r)
    {
        longdouble tmp;
        tmp.init(*this).value()->divide(*r.value(), llvm::APFloat::rmNearestTiesToEven);
        return tmp;
    }

    longdouble operator%(const longdouble &r)
    {
        longdouble tmp;
        tmp.init(*this).value()->mod(*r.value(), llvm::APFloat::rmNearestTiesToEven);
        return tmp;
    }

    bool operator<(const longdouble &r)
    {
        return value()->compare(*r.value()) == llvm::APFloat::cmpLessThan;
    }

    bool operator>(const longdouble &r)
    {
        return value()->compare(*r.value()) == llvm::APFloat::cmpGreaterThan;
    }

    bool operator<=(const longdouble &r)
    {
        llvm::APFloat::cmpResult res = value()->compare(*r.value());
        return res == llvm::APFloat::cmpLessThan
               || res == llvm::APFloat::cmpEqual;
    }

    bool operator>=(const longdouble &r)
    {
        llvm::APFloat::cmpResult res = value()->compare(*r.value());
        return res == llvm::APFloat::cmpGreaterThan
               || res == llvm::APFloat::cmpEqual;
    }

    bool operator==(const longdouble &r)
    {
        return value()->compare(*r.value()) == llvm::APFloat::cmpEqual;
    }

    bool operator!=(const longdouble &r)
    {
        llvm::APFloat::cmpResult res = value()->compare(*r.value());
        return res != llvm::APFloat::cmpEqual;
    }

    operator float () const
    {
        return convertToFloat();
    }
    operator double () const
    {
        return convertToDouble();
    }

    operator signed char       () const { return convertToInteger<signed char, false>(); }
    operator short             () const { return convertToInteger<short, false>(); }
    operator int               () const { return convertToInteger<int, false>(); }
    operator long              () const { return convertToInteger<long, false>(); }
    operator long long         () const { return convertToInteger<long long, false>(); }

    operator unsigned char     () const { return convertToInteger<unsigned char, true>(); }
    operator unsigned short    () const { return convertToInteger<unsigned short, true>(); }
    operator unsigned int      () const { return convertToInteger<unsigned int, true>(); }
    operator unsigned long     () const { return convertToInteger<unsigned long, true>(); }
    operator unsigned long long() const { return convertToInteger<unsigned long long, true>(); }

    operator llvm::APFloat& ()
    {
        return *value();
    }

     operator const llvm::APFloat& () const
    {
        return *value();
    }

    operator bool () const
    {
        return value()->isZero() ? false : true;
    }

    double convertToDouble() const
    {
        llvm::APFloat trunc(*value());
        bool ignored;
        llvm::APFloat::opStatus status = trunc.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmNearestTiesToEven, &ignored);
        //assert(status == llvm::APFloat::opOK);
        (void)status;
        return trunc.convertToDouble();
    }

    float convertToFloat() const
    {
        llvm::APFloat trunc(*value());
        bool ignored;
        llvm::APFloat::opStatus status = trunc.convert(llvm::APFloat::IEEEsingle, llvm::APFloat::rmNearestTiesToEven, &ignored);
        //assert(status == llvm::APFloat::opOK);
        (void)status;
        return trunc.convertToFloat();
    }

    template<typename T, bool IsUnsigned>
    T convertToInteger() const
    {
        llvm::APSInt val(8*sizeof(T), IsUnsigned);
        bool ignored;
        value()->convertToInteger(val, llvm::APFloat::rmNearestTiesToEven, &ignored);
        return static_cast<T>(IsUnsigned ? val.getZExtValue() : val.getSExtValue());
    }

    longdouble &set(int8_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, true, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(int16_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, true, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(int32_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, true, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(int64_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, true, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(uint8_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, false, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(uint16_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, false, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(uint32_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, false, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    longdouble &set(uint64_t i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, false, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }

    // Apple sdk on osx defines uint64_t as unsigned long long, so long types
    // are also needed to play with mars.h typedefs (dinteger_t and sinteger_t).
#if defined(__APPLE__) && defined(__LP64__)
    longdouble &set(long i) { return set((int64_t)i); }
    longdouble &set(unsigned long i) { return set((uint64_t)i); }
#endif

#if defined(_MSC_VER)
    longdouble &set(unsigned long i)
    {
        const llvm::integerPart tmp = static_cast<llvm::integerPart>(i);
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromSignExtendedInteger(&tmp, 1, false, llvm::APFloat::rmNearestTiesToEven);
        return init(v);
    }
#endif

    longdouble &set(float r)
    {
        llvm::APFloat extended(r);
        bool ignored;
        llvm::APFloat::opStatus status = extended.convert(getFltSemantics(), llvm::APFloat::rmNearestTiesToEven, &ignored);
        assert(status == llvm::APFloat::opOK);
        (void)status;
        return init(extended);
    }

    longdouble &set(double r)
    {
        llvm::APFloat extended(r);
        bool ignored;
        llvm::APFloat::opStatus status = extended.convert(getFltSemantics(), llvm::APFloat::rmNearestTiesToEven, &ignored);
        assert(status == llvm::APFloat::opOK);
        (void)status;
        return init(extended);
    }

    longdouble &set(const llvm::APFloat &r)
    {
        return init(r);
    }

    longdouble &set(const longdouble &r)
    {
        return init(r);
    }

    bool isNaN() const
    {
        return value()->isNaN();
    }

    bool isSignaling() const
    {
#if LDC_LLVM_VER >= 304
        return value()->isSignaling();
#else
        if (!value()->isNaN())
            return false;
        double d = convertToDouble();
        /* A signalling NaN is a NaN with 0 as the most significant bit of
         * its significand, which is bit 51 of 0..63 for 64 bit doubles.
         * FIXME: Check: meaning of the bit is reversed for MIPS?!?!
         */
        return !((((unsigned char*)&d)[6]) & 8);
#endif
    }

    static longdouble getNaN()
    {
        longdouble tmp;
        return tmp.init(llvm::APFloat::getNaN(getFltSemantics()));
    }

    static longdouble getSNaN()
    {
        longdouble tmp;
        return tmp.init(llvm::APFloat::getSNaN(getFltSemantics()));
    }

    static longdouble getInf()
    {
        longdouble tmp;
        return tmp.init(llvm::APFloat::getInf(getFltSemantics()));
    }

    static longdouble getLargest()
    {
        longdouble tmp;
        return tmp.init(llvm::APFloat::getLargest(getFltSemantics()));
    }

    longdouble abs() const;
    longdouble sqrt() const;
    longdouble sin() const;
    longdouble cos() const;
    longdouble tan() const;

    longdouble floor() const;
    longdouble ceil() const;
    longdouble trunc() const;
    longdouble round() const;

    static longdouble fmin(longdouble x, longdouble y);
    static longdouble fmax(longdouble x, longdouble y);

    static longdouble fmod(longdouble x, longdouble y);
    static longdouble ldexp(longdouble ldval, int exp);

    static longdouble convertFromString(const char *str)
    {
        llvm::APFloat v(getFltSemantics(), llvm::APFloat::uninitialized);
        v.convertFromString(llvm::StringRef(str), llvm::APFloat::rmNearestTiesToEven);
        longdouble tmp;
        return tmp.init(v);
    }

    static bool fequal(const longdouble &x, const longdouble &y)
    {
        return x.value()->bitwiseIsEqual(*(y.value()));
    }

    int format(char *buf) const;
    int formatHex(char *buf, bool upper) const;
};

} // namespace ldc

typedef ldc::longdouble longdouble;
typedef ldc::longdouble volatile_longdouble;

// Use ldouble() to explicitely create a longdouble value.
template<typename T>
inline longdouble
ldouble (T x)
{
    longdouble d;
    d.set(x);
    return d;
}

template<typename T> inline longdouble operator+(longdouble ld, T x) { return ld + ldouble(x); }
template<typename T> inline longdouble operator-(longdouble ld, T x) { return ld - ldouble(x); }
template<typename T> inline longdouble operator*(longdouble ld, T x) { return ld * ldouble(x); }
template<typename T> inline longdouble operator/(longdouble ld, T x) { return ld / ldouble(x); }

template<typename T> inline longdouble operator+(T x, longdouble ld) { return ldouble(x) + ld; }
template<typename T> inline longdouble operator-(T x, longdouble ld) { return ldouble(x) - ld; }
template<typename T> inline longdouble operator*(T x, longdouble ld) { return ldouble(x) * ld; }
template<typename T> inline longdouble operator/(T x, longdouble ld) { return ldouble(x) / ld; }

template<typename T> inline longdouble& operator+=(longdouble& ld, T x) { return ld = ld + x; }
template<typename T> inline longdouble& operator-=(longdouble& ld, T x) { return ld = ld - x; }
template<typename T> inline longdouble& operator*=(longdouble& ld, T x) { return ld = ld * x; }
template<typename T> inline longdouble& operator/=(longdouble& ld, T x) { return ld = ld / x; }

template<typename T> inline bool operator< (longdouble ld, T x) { return ld <  ldouble(x); }
template<typename T> inline bool operator<=(longdouble ld, T x) { return ld <= ldouble(x); }
template<typename T> inline bool operator> (longdouble ld, T x) { return ld >  ldouble(x); }
template<typename T> inline bool operator>=(longdouble ld, T x) { return ld >= ldouble(x); }
template<typename T> inline bool operator==(longdouble ld, T x) { return ld == ldouble(x); }
template<typename T> inline bool operator!=(longdouble ld, T x) { return ld != ldouble(x); }

template<typename T> inline bool operator< (T x, longdouble ld) { return ldouble(x) <  ld; }
template<typename T> inline bool operator<=(T x, longdouble ld) { return ldouble(x) <= ld; }
template<typename T> inline bool operator> (T x, longdouble ld) { return ldouble(x) >  ld; }
template<typename T> inline bool operator>=(T x, longdouble ld) { return ldouble(x) >= ld; }
template<typename T> inline bool operator==(T x, longdouble ld) { return ldouble(x) == ld; }
template<typename T> inline bool operator!=(T x, longdouble ld) { return ldouble(x) != ld; }

inline longdouble fabsl(longdouble ld) { return ld.abs(); }
inline longdouble sqrtl(longdouble ld) { return ld.sqrt(); }
inline longdouble sinl (longdouble ld) { return ld.sin(); }
inline longdouble cosl (longdouble ld) { return ld.cos(); }
inline longdouble tanl (longdouble ld) { return ld.tan(); }
inline longdouble floorl (longdouble ld) { return ld.floor(); }
inline longdouble ceill (longdouble ld) { return ld.ceil(); }
inline longdouble truncl (longdouble ld) { return ld.trunc(); }
inline longdouble roundl (longdouble ld) { return ld.round(); }

inline longdouble fminl(longdouble x, longdouble y) { return ldc::longdouble::fmin(x, y); }
inline longdouble fmaxl(longdouble x, longdouble y) { return ldc::longdouble::fmax(x, y); }
inline longdouble fmodl(longdouble x, longdouble y) { return ldc::longdouble::fmod(x, y); }
inline longdouble ldexpl(longdouble ldval, int exp) { return ldc::longdouble::ldexp(ldval, exp); }

inline longdouble fabs (longdouble ld) { return fabsl(ld); }
inline longdouble sqrt (longdouble ld) { return sqrtl(ld); }
inline longdouble floor (longdouble ld) { return floorl(ld); }
inline longdouble ceil (longdouble ld) { return ceill(ld); }
inline longdouble trunc (longdouble ld) { return truncl(ld); }
inline longdouble round (longdouble ld) { return roundl(ld); }
inline longdouble fmin(longdouble x, longdouble y) { return fminl(x,y); }
inline longdouble fmax(longdouble x, longdouble y) { return fmaxl(x,y); }

inline size_t
ld_sprint(char* str, int fmt, longdouble x)
{
    // The signature of this method leads to buffer overflows.
    if (fmt == 'a' || fmt == 'A')
       return x.formatHex(str, fmt == 'A');
    assert(fmt == 'g');
    return x.format(str);
}

namespace ldc
{

// List of values for .max, .min, etc, for floats in D.
struct real_properties
{
    longdouble maxval, minval, epsilonval;
    int64_t dig, mant_dig;
    int64_t max_10_exp, min_10_exp;
    int64_t max_exp, min_exp;
};

extern real_properties real_limits[longdouble::NumModes];

// Initialize real_properties.
void real_init();

} // namespace ldc

// Macros are used by the D frontend, so map to longdouble property values instead of host long double.
#undef FLT_MAX
#undef DBL_MAX
#undef LDBL_MAX
#undef FLT_MIN
#undef DBL_MIN
#undef LDBL_MIN
#undef FLT_DIG
#undef DBL_DIG
#undef LDBL_DIG
#undef FLT_MANT_DIG
#undef DBL_MANT_DIG
#undef LDBL_MANT_DIG
#undef FLT_MAX_10_EXP
#undef DBL_MAX_10_EXP
#undef LDBL_MAX_10_EXP
#undef FLT_MIN_10_EXP
#undef DBL_MIN_10_EXP
#undef LDBL_MIN_10_EXP
#undef FLT_MAX_EXP
#undef DBL_MAX_EXP
#undef LDBL_MAX_EXP
#undef FLT_MIN_EXP
#undef DBL_MIN_EXP
#undef LDBL_MIN_EXP
#undef FLT_EPSILON
#undef DBL_EPSILON
#undef LDBL_EPSILON

#define FLT_MAX ldc::real_limits[ldc::longdouble::Float].maxval;
#define DBL_MAX ldc::real_limits[ldc::longdouble::Double].maxval;
#define LDBL_MAX ldc::real_limits[ldc::longdouble::LongDouble].maxval;
#define FLT_MIN ldc::real_limits[ldc::longdouble::Float].minval;
#define DBL_MIN ldc::real_limits[ldc::longdouble::Double].minval;
#define LDBL_MIN ldc::real_limits[ldc::longdouble::LongDouble].minval;
#define FLT_DIG ldc::real_limits[ldc::longdouble::Float].dig;
#define DBL_DIG ldc::real_limits[ldc::longdouble::Double].dig;
#define LDBL_DIG ldc::real_limits[ldc::longdouble::LongDouble].dig;
#define FLT_MANT_DIG ldc::real_limits[ldc::longdouble::Float].mant_dig;
#define DBL_MANT_DIG ldc::real_limits[ldc::longdouble::Double].mant_dig;
#define LDBL_MANT_DIG ldc::real_limits[ldc::longdouble::LongDouble].mant_dig;
#define FLT_MAX_10_EXP ldc::real_limits[ldc::longdouble::Float].max_10_exp;
#define DBL_MAX_10_EXP ldc::real_limits[ldc::longdouble::Double].max_10_exp;
#define LDBL_MAX_10_EXP ldc::real_limits[ldc::longdouble::LongDouble].max_10_exp;
#define FLT_MIN_10_EXP ldc::real_limits[ldc::longdouble::Float].min_10_exp;
#define DBL_MIN_10_EXP ldc::real_limits[ldc::longdouble::Double].min_10_exp;
#define LDBL_MIN_10_EXP ldc::real_limits[ldc::longdouble::LongDouble].min_10_exp;
#define FLT_MAX_EXP ldc::real_limits[ldc::longdouble::Float].max_exp;
#define DBL_MAX_EXP ldc::real_limits[ldc::longdouble::Double].max_exp;
#define LDBL_MAX_EXP ldc::real_limits[ldc::longdouble::LongDouble].max_exp;
#define FLT_MIN_EXP ldc::real_limits[ldc::longdouble::Float].min_exp;
#define DBL_MIN_EXP ldc::real_limits[ldc::longdouble::Double].min_exp;
#define LDBL_MIN_EXP ldc::real_limits[ldc::longdouble::LongDouble].min_exp;
#define FLT_EPSILON ldc::real_limits[ldc::longdouble::Float].epsilonval;
#define DBL_EPSILON ldc::real_limits[ldc::longdouble::Double].epsilonval;
#define LDBL_EPSILON ldc::real_limits[ldc::longdouble::LongDouble].epsilonval;

#endif
