//===-- real_t.d ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module ddmd.root.real_t;

private struct APFloatPrototype
{
    void* semantics;
    union
    {
        long part;
        long* parts;
    }
    short exponent;
    int categoryAndSign;
}

extern (C++, ldc) struct real_t
{
    this(float f) { initFrom(f); }
    this(double f) { initFrom(f); }
    this(int i) { initFrom(i); }
    this(long i) { initFrom(i); }
    this(uint i) { initFrom(i); }
    this(ulong i) { initFrom(i); }

    bool toBool() const;
    float toFloat32() const;
    double toFloat64() const;
    int toInt32() const;
    long toInt64() const;
    uint toUInt32() const;
    ulong toUInt64() const;

    extern(D) T opCast(T)() const
    {
             static if (is(T == bool))   return toBool();
        else static if (is(T == float))  return toFloat();
        else static if (is(T == double)) return toDouble();
        else static if (is(T == int))    return toInt32();
        else static if (is(T == long))   return toInt64();
        else static if (is(T == uint))   return toUInt32();
        else static if (is(T == ulong))  return toUInt64();
        else static assert(0, "Trying to cast real_t to unsupported type");
    }

    // arithmetic operators
    real_t opNeg() const;
    real_t add(const ref real_t r) const;
    real_t sub(const ref real_t r) const;
    real_t mul(const ref real_t r) const;
    real_t div(const ref real_t r) const;
    real_t mod(const ref real_t r) const;

    // D binary operators (for rvalues)
    extern(D) real_t opBinary(string op)(real_t r) const
    {
             static if (op == "+") return add(r);
        else static if (op == "-") return sub(r);
        else static if (op == "*") return mul(r);
        else static if (op == "/") return div(r);
        else static if (op == "%") return mod(r);
        else static assert(0, "Unsupported binary operator");
    }

    // comparison
    int cmp(const ref real_t r) const;

    extern(D) int opCmp(real_t r) const { return cmp(r); }
    extern(D) bool opEquals(real_t r) const { return cmp(r) == 0; }

    // D lifetime
    extern(D) this(this) { postblit(); }
    extern(D) void opAssign(real_t r) { moveAssign(r); }
    extern(D) ~this() { destruct(); }

private:
    // non-trivial llvm::APFloat, the hard way:
    union
    {
        byte[APFloatPrototype.sizeof] value = void;
        long for_alignment_only = void;

        // Designate default-constructed real_t instances in D (with invalid
        // APFloat values) by initializing the pointer at the beginning with
        // null.
        // See comment in C++ header.
        void* valueSemantics = null;
    }

    void initFrom(float f);
    void initFrom(double f);
    void initFrom(int i);
    void initFrom(long i);
    void initFrom(uint i);
    void initFrom(ulong i);

    bool isInitialized() const;
    void safeInit();
    void postblit();
    void moveAssign(ref real_t r);
    void destruct();
}
