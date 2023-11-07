/**
 * Implement CTFE for intrinsic (builtin) functions.
 *
 * Currently includes functions from `std.math`, `core.math` and `core.bitop`.
 *
 * Copyright:   Copyright (C) 1999-2023 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 https://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 https://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/builtin.d, _builtin.d)
 * Documentation:  https://dlang.org/phobos/dmd_builtin.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/builtin.d
 */

module dmd.builtin;

import dmd.arraytypes;
import dmd.astenums;
import dmd.dmangle;
import dmd.errors;
import dmd.expression;
import dmd.func;
import dmd.globals;
import dmd.location;
import dmd.mtype;
import dmd.root.ctfloat;
import dmd.tokens;
import dmd.id;
static import core.bitop;

/**********************************
 * Determine if function is a builtin one that we can
 * evaluate at compile time.
 */
public extern (C++) BUILTIN isBuiltin(FuncDeclaration fd)
{
    if (fd.builtin == BUILTIN.unknown)
    {
        fd.builtin = determine_builtin(fd);
    }
    return fd.builtin;
}

/**************************************
 * Evaluate builtin function.
 * Return result; NULL if cannot evaluate it.
 */
public extern (C++) Expression eval_builtin(const ref Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    if (fd.builtin == BUILTIN.unimp)
        return null;

    switch (fd.builtin)
    {
        foreach(e; __traits(allMembers, BUILTIN))
        {
            static if (e == "unknown")
                case BUILTIN.unknown: assert(false);
            else static if (IN_LLVM && e.length > 5 && e[0..5] == "llvm_")
                mixin("case BUILTIN."~e~": return eval_llvm"~e[5..$]~"(loc, fd, arguments);");
            else
                mixin("case BUILTIN."~e~": return eval_"~e~"(loc, fd, arguments);");
        }
        default: assert(0);
    }
}

private:

/**
 * Handler for evaluating builtins during CTFE.
 *
 * Params:
 *  loc = The call location, for error reporting.
 *  fd = The callee declaration, e.g. to disambiguate between different overloads
 *       in a single handler (LDC).
 *  arguments = The function call arguments.
 * Returns:
 *  An Expression containing the return value of the call.
 */

BUILTIN determine_builtin(FuncDeclaration func)
{
    auto fd = func.toAliasFunc();
    if (fd.isDeprecated())
        return BUILTIN.unimp;

version (IN_LLVM)
{
    import dmd.root.string : toDString;
    import gen.dpragma : LDCPragma;

    if (func.llvmInternal == LDCPragma.LLVMintrinsic)
    {
        const(char)[] name = func.intrinsicName.toDString;
        if (name.length < 7 || name[0..5] != "llvm.")
            return BUILTIN.unimp;

        // find next "." after "llvm."
        size_t end = 0;
        foreach (i; 6 .. name.length)
        {
            if (name[i] == '.')
            {
                end = i;
                break;
            }
        }
        if (end == 0)
            return BUILTIN.unimp;

        name = name[5 .. end]; // e.g., "llvm.sin.f32" => "sin"
        switch (name)
        {
            foreach (e; __traits(allMembers, BUILTIN))
            {
                static if (e.length > 5 && e[0..5] == "llvm_")
                    mixin(`case "`~e[5..$]~`": return BUILTIN.`~e~";");
            }
            default: return BUILTIN.unimp;
        }
    }
}

    auto m = fd.getModule();
    if (!m || !m.md)
        return BUILTIN.unimp;
    const md = m.md;

    // Look for core.math, core.bitop, std.math, and std.math.<package>
    const id2 = (md.packages.length == 2) ? md.packages[1] : md.id;
    if (id2 != Id.math && id2 != Id.bitop)
        return BUILTIN.unimp;

    if (md.packages.length != 1 && !(md.packages.length == 2 && id2 == Id.math))
        return BUILTIN.unimp;

    const id1 = md.packages[0];
    if (id1 != Id.core && id1 != Id.std)
        return BUILTIN.unimp;
    const id3 = fd.ident;

    if (id1 == Id.core && id2 == Id.bitop)
    {
        if (id3 == Id.bsf)     return BUILTIN.bsf;
        if (id3 == Id.bsr)     return BUILTIN.bsr;
        if (id3 == Id.bswap)   return BUILTIN.bswap;
        if (id3 == Id._popcnt) return BUILTIN.popcnt;
        return BUILTIN.unimp;
    }

    // Math
    if (id3 == Id.sin)   return BUILTIN.sin;
    if (id3 == Id.cos)   return BUILTIN.cos;
    if (id3 == Id.tan)   return BUILTIN.tan;
    if (id3 == Id.atan2) return BUILTIN.unimp; // N.B unimplmeneted

    if (id3 == Id._sqrt) return BUILTIN.sqrt;
    if (id3 == Id.fabs)  return BUILTIN.fabs;

    if (id3 == Id.exp)    return BUILTIN.exp;
    if (id3 == Id.expm1)  return BUILTIN.expm1;
    if (id3 == Id.exp2)   return BUILTIN.exp2;
version (IN_LLVM)
{
    // Our implementations in CTFloat fall back to a generic version in case
    // host compiler's druntime doesn't provide core.math.yl2x[p1] (GDC,
    // non-x86 hosts). Not providing yl2x[p1] for CTFE would significantly
    // limit CTFE-ability of std.math for x86 targets.
    if (id3 == Id.yl2x)   return BUILTIN.yl2x;
    if (id3 == Id.yl2xp1) return BUILTIN.yl2xp1;
}
else
{
    if (id3 == Id.yl2x)   return CTFloat.yl2x_supported ? BUILTIN.yl2x : BUILTIN.unimp;
    if (id3 == Id.yl2xp1) return CTFloat.yl2xp1_supported ? BUILTIN.yl2xp1 : BUILTIN.unimp;
}

    if (id3 == Id.log)   return BUILTIN.log;
    if (id3 == Id.log2)  return BUILTIN.log2;
    if (id3 == Id.log10) return BUILTIN.log10;

    if (id3 == Id.ldexp) return BUILTIN.ldexp;
    if (id3 == Id.round) return BUILTIN.round;
    if (id3 == Id.floor) return BUILTIN.floor;
    if (id3 == Id.ceil)  return BUILTIN.ceil;
    if (id3 == Id.trunc) return BUILTIN.trunc;

    if (id3 == Id.fmin)     return BUILTIN.fmin;
    if (id3 == Id.fmax)     return BUILTIN.fmax;
    if (id3 == Id.fma)      return BUILTIN.fma;
    if (id3 == Id.copysign) return BUILTIN.copysign;

    if (id3 == Id.isnan)      return BUILTIN.isnan;
    if (id3 == Id.isInfinity) return BUILTIN.isinfinity;
    if (id3 == Id.isfinite)   return BUILTIN.isfinite;

    // Only match pow(fp,fp) where fp is a floating point type
    if (id3 == Id._pow)
    {
        if ((*fd.parameters)[0].type.isfloating() &&
            (*fd.parameters)[1].type.isfloating())
            return BUILTIN.pow;
        return BUILTIN.unimp;
    }

    if (id3 != Id.toPrec)
        return BUILTIN.unimp;
    const(char)* me = mangleExact(fd);
    final switch (me["_D4core4math__T6toPrecHT".length])
    {
        case 'd': return BUILTIN.toPrecDouble;
        case 'e': return BUILTIN.toPrecReal;
        case 'f': return BUILTIN.toPrecFloat;
    }
}

Expression eval_unimp(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    return null;
}

Expression eval_sin(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.sin(arg0.toReal()), arg0.type);
}

Expression eval_cos(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.cos(arg0.toReal()), arg0.type);
}

Expression eval_tan(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.tan(arg0.toReal()), arg0.type);
}

Expression eval_sqrt(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.sqrt(arg0.toReal()), arg0.type);
}

Expression eval_fabs(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.fabs(arg0.toReal()), arg0.type);
}

Expression eval_ldexp(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.int64);
    return new RealExp(loc, CTFloat.ldexp(arg0.toReal(), cast(int) arg1.toInteger()), arg0.type);
}

Expression eval_log(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log(arg0.toReal()), arg0.type);
}

Expression eval_log2(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log2(arg0.toReal()), arg0.type);
}

Expression eval_log10(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log10(arg0.toReal()), arg0.type);
}

Expression eval_exp(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.exp(arg0.toReal()), arg0.type);
}

Expression eval_expm1(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.expm1(arg0.toReal()), arg0.type);
}

Expression eval_exp2(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.exp2(arg0.toReal()), arg0.type);
}

Expression eval_round(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.round(arg0.toReal()), arg0.type);
}

Expression eval_floor(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.floor(arg0.toReal()), arg0.type);
}

Expression eval_ceil(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.ceil(arg0.toReal()), arg0.type);
}

Expression eval_trunc(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.trunc(arg0.toReal()), arg0.type);
}

Expression eval_copysign(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.copysign(arg0.toReal(), arg1.toReal()), arg0.type);
}

Expression eval_pow(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.pow(arg0.toReal(), arg1.toReal()), arg0.type);
}

Expression eval_fmin(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.fmin(arg0.toReal(), arg1.toReal()), arg0.type);
}

Expression eval_fmax(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.fmax(arg0.toReal(), arg1.toReal()), arg0.type);
}

Expression eval_fma(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    Expression arg2 = (*arguments)[2];
    assert(arg2.op == EXP.float64);
    return new RealExp(loc, CTFloat.fma(arg0.toReal(), arg1.toReal(), arg2.toReal()), arg0.type);
}

Expression eval_isnan(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return IntegerExp.createBool(CTFloat.isNaN(arg0.toReal()));
}

Expression eval_isinfinity(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return IntegerExp.createBool(CTFloat.isInfinity(arg0.toReal()));
}

Expression eval_isfinite(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    const value = !CTFloat.isNaN(arg0.toReal()) && !CTFloat.isInfinity(arg0.toReal());
    return IntegerExp.createBool(value);
}

Expression eval_bsf(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    if (n == 0)
        error(loc, "`bsf(0)` is undefined");
    return new IntegerExp(loc, core.bitop.bsf(n), Type.tint32);
}

Expression eval_bsr(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    if (n == 0)
        error(loc, "`bsr(0)` is undefined");
    return new IntegerExp(loc, core.bitop.bsr(n), Type.tint32);
}

Expression eval_bswap(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    TY ty = arg0.type.toBasetype().ty;
    if (ty == Tint64 || ty == Tuns64)
        return new IntegerExp(loc, core.bitop.bswap(cast(ulong) n), arg0.type);
    else
        return new IntegerExp(loc, core.bitop.bswap(cast(uint) n), arg0.type);
}

Expression eval_popcnt(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    return new IntegerExp(loc, core.bitop.popcnt(n), Type.tint32);
}

Expression eval_yl2x(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    const x = arg0.toReal();
    const y = arg1.toReal();
    real_t result = CTFloat.zero;
    CTFloat.yl2x(&x, &y, &result);
    return new RealExp(loc, result, arg0.type);
}

Expression eval_yl2xp1(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    const x = arg0.toReal();
    const y = arg1.toReal();
    real_t result = CTFloat.zero;
    CTFloat.yl2xp1(&x, &y, &result);
    return new RealExp(loc, result, arg0.type);
}

Expression eval_toPrecFloat(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    float f = cast(real)arg0.toReal();
    return new RealExp(loc, real_t(f), Type.tfloat32);
}

Expression eval_toPrecDouble(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    double d = cast(real)arg0.toReal();
    return new RealExp(loc, real_t(d), Type.tfloat64);
}

Expression eval_toPrecReal(Loc loc, FuncDeclaration fd, Expressions* arguments)
{
    Expression arg0 = (*arguments)[0];
    return new RealExp(loc, arg0.toReal(), Type.tfloat80);
}

// These built-ins are reserved for GDC and LDC.
Expression eval_gcc(Loc, FuncDeclaration, Expressions*)
{
    assert(0);
}

Expression eval_llvm(Loc, FuncDeclaration, Expressions*)
{
    assert(0);
}

version (IN_LLVM)
{

private Type getTypeOfOverloadedIntrinsic(FuncDeclaration fd)
{
    import dmd.dtemplate : TemplateInstance;

    // Depending on the state of the code generation we have to look at
    // the template instance or the function declaration.
    assert(fd.parent && "function declaration requires parent");
    TemplateInstance tinst = fd.parent.isTemplateInstance();
    if (tinst)
    {
        // See DtoOverloadedIntrinsicName
        assert(tinst.tdtypes.length == 1);
        return cast(Type) tinst.tdtypes[0];
    }
    else
    {
        assert(fd.type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction) fd.type;
        assert(tf.parameterList.length >= 1);
        return (*tf.parameterList.parameters)[0].type;
    }
}

private int getBitsizeOfType(Loc loc, Type type)
{
    switch (type.toBasetype().ty)
    {
      case Tint64:
      case Tuns64: return 64;
      case Tint32:
      case Tuns32: return 32;
      case Tint16:
      case Tuns16: return 16;
      case Tint128:
      case Tuns128:
          error(loc, "cent/ucent not supported");
          break;
      default:
          error(loc, "unsupported type");
          break;
    }
    return 32; // in case of error
}

Expression eval_llvmsin(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.sin(arg0.toReal()), type);
}

Expression eval_llvmcos(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.cos(arg0.toReal()), type);
}

Expression eval_llvmsqrt(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.sqrt(arg0.toReal()), type);
}

Expression eval_llvmexp(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.exp(arg0.toReal()), type);
}

Expression eval_llvmexp2(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.exp2(arg0.toReal()), type);
}

Expression eval_llvmlog(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log(arg0.toReal()), type);
}

Expression eval_llvmlog2(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log2(arg0.toReal()), type);
}

Expression eval_llvmlog10(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.log10(arg0.toReal()), type);
}

Expression eval_llvmfabs(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.fabs(arg0.toReal()), type);
}

Expression eval_llvmminnum(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.fmin(arg0.toReal(), arg1.toReal()), type);
}

Expression eval_llvmmaxnum(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.fmax(arg0.toReal(), arg1.toReal()), type);
}

Expression eval_llvmfloor(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.floor(arg0.toReal()), type);
}

Expression eval_llvmceil(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.ceil(arg0.toReal()), type);
}

Expression eval_llvmtrunc(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.trunc(arg0.toReal()), type);
}

Expression eval_llvmrint(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.rint(arg0.toReal()), type);
}

Expression eval_llvmnearbyint(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.nearbyint(arg0.toReal()), type);
}

Expression eval_llvmround(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    return new RealExp(loc, CTFloat.round(arg0.toReal()), type);
}

Expression eval_llvmfma(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    Expression arg2 = (*arguments)[2];
    assert(arg2.op == EXP.float64);
    return new RealExp(loc, CTFloat.fma(arg0.toReal(), arg1.toReal(), arg2.toReal()), type);
}

Expression eval_llvmcopysign(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);
    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.float64);
    Expression arg1 = (*arguments)[1];
    assert(arg1.op == EXP.float64);
    return new RealExp(loc, CTFloat.copysign(arg0.toReal(), arg1.toReal()), type);
}

Expression eval_llvmbswap(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);

    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    enum ulong BYTEMASK = 0x00FF00FF00FF00FF;
    enum ulong SHORTMASK = 0x0000FFFF0000FFFF;
    enum ulong INTMASK = 0x00000000FFFFFFFF;
    switch (type.toBasetype().ty)
    {
      case Tint64:
      case Tuns64:
          // swap high and low uints
          n = ((n >> 32) & INTMASK) | ((n & INTMASK) << 32);
          goto case Tuns32;
      case Tint32:
      case Tuns32:
          // swap adjacent ushorts
          n = ((n >> 16) & SHORTMASK) | ((n & SHORTMASK) << 16);
          goto case Tuns16;
      case Tint16:
      case Tuns16:
          // swap adjacent ubytes
          n = ((n >> 8 ) & BYTEMASK)  | ((n & BYTEMASK) << 8 );
          break;
      case Tint128:
      case Tuns128:
          error(loc, "cent/ucent not supported");
          break;
      default:
          error(loc, "unsupported type");
          break;
    }
    return new IntegerExp(loc, n, type);
}

Expression eval_llvmcttz(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);

    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t x = arg0.toInteger();

    int n = getBitsizeOfType(loc, type);

    if (x == 0)
    {
        if ((*arguments)[1].toInteger())
            error(loc, "llvm.cttz.i#(0) is undefined");
    }
    else
    {
        int c = n >> 1;
        n -= 1;
        const uinteger_t mask = (uinteger_t(1L) << n) | (uinteger_t(1L) << n)-1;
        do {
            uinteger_t y = (x << c) & mask;
            if (y != 0) { n -= c; x = y; }
            c = c >> 1;
        } while (c != 0);
    }

    return new IntegerExp(loc, n, type);
}

Expression eval_llvmctlz(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);

    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t x = arg0.toInteger();
    if (x == 0 && (*arguments)[1].toInteger())
        error(loc, "llvm.ctlz.i#(0) is undefined");

    int n = getBitsizeOfType(loc, type);
    int c = n >> 1;
    do {
        uinteger_t y = x >> c; if (y != 0) { n -= c; x = y; }
        c = c >> 1;
    } while (c != 0);

    return new IntegerExp(loc, n - x, type);
}

Expression eval_llvmctpop(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    // FIXME Does not work for cent/ucent
    Type type = getTypeOfOverloadedIntrinsic(fd);

    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);
    uinteger_t n = arg0.toInteger();
    int cnt = 0;
    while (n)
    {
        cnt += (n & 1);
        n >>= 1;
    }
    return new IntegerExp(loc, cnt, type);
}

Expression eval_llvmexpect(Loc loc, FuncDeclaration fd, Expressions *arguments)
{
    Type type = getTypeOfOverloadedIntrinsic(fd);

    Expression arg0 = (*arguments)[0];
    assert(arg0.op == EXP.int64);

    return new IntegerExp(loc, arg0.toInteger(), type);
}

} // IN_LLVM
