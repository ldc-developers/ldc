
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/builtin.c
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>                     // strcmp()
#include <math.h>

#include "mars.h"
#include "declaration.h"
#include "attrib.h"
#include "expression.h"
#include "scope.h"
#include "mtype.h"
#include "aggregate.h"
#include "identifier.h"
#include "id.h"
#include "module.h"
#include "root/port.h"
#include "tokens.h"
#if IN_LLVM
#include "template.h"
#include "gen/pragma.h"
#endif

StringTable builtins;

void add_builtin(const char *mangle, builtin_fp fp)
{
    builtins.insert(mangle, strlen(mangle))->ptrvalue = (void *)fp;
}

builtin_fp builtin_lookup(const char *mangle)
{
    if (StringValue *sv = builtins.lookup(mangle, strlen(mangle)))
        return (builtin_fp)sv->ptrvalue;
    return NULL;
}

Expression *eval_unimp(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    return NULL;
}

Expression *eval_sin(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, sinl(arg0->toReal()), arg0->type);
}

Expression *eval_cos(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, cosl(arg0->toReal()), arg0->type);
}

Expression *eval_tan(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, tanl(arg0->toReal()), arg0->type);
}

Expression *eval_sqrt(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, Port::sqrt(arg0->toReal()), arg0->type);
}

Expression *eval_fabs(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, fabsl(arg0->toReal()), arg0->type);
}

#if IN_LLVM

static inline Type *getTypeOfOverloadedIntrinsic(FuncDeclaration *fd)
{
    // Depending on the state of the code generation we have to look at
    // the template instance or the function declaration.
    assert(fd->parent && "function declaration requires parent");
    TemplateInstance* tinst = fd->parent->isTemplateInstance();
    if (tinst)
    {
        // See DtoOverloadedIntrinsicName
        assert(tinst->tdtypes.dim == 1);
        return static_cast<Type*>(tinst->tdtypes.data[0]);
    }
    else
    {
        assert(fd->type->ty == Tfunction);
        TypeFunction *tf = static_cast<TypeFunction *>(fd->type);
        assert(tf->parameters->dim >= 1);
        return tf->parameters->data[0]->type;
    }
}

static inline int getBitsizeOfType(Loc loc, Type *type)
{
    switch (type->toBasetype()->ty)
    {
      case Tint64:
      case Tuns64: return 64;
      case Tint32:
      case Tuns32: return 32;
      case Tint16:
      case Tuns16: return 16;
      case Tint128:
      case Tuns128:
#if WANT_CENT
        return 128;
#else
          error(loc, "cent/ucent not supported");
          break;
#endif
      default:
          error(loc, "unsupported type");
          break;
    }
    return 32; // in case of error
}

Expression *eval_llvmsin(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, sinl(arg0->toReal()), type);
}

Expression *eval_llvmcos(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, cosl(arg0->toReal()), type);
}

Expression *eval_llvmsqrt(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, sqrtl(arg0->toReal()), type);
}

Expression *eval_llvmfabs(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, fabsl(arg0->toReal()), type);
}

Expression *eval_llvmminnum(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    Expression *arg1 = (*arguments)[1];
    assert(arg1->op == TOKfloat64);
    return new RealExp(loc, fminl(arg0->toReal(), arg1->toReal()), type);
}

Expression *eval_llvmmaxnum(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    Expression *arg1 = (*arguments)[1];
    assert(arg1->op == TOKfloat64);
    return new RealExp(loc, fmaxl(arg0->toReal(), arg1->toReal()), type);
}

Expression *eval_llvmfloor(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, floor(arg0->toReal()), type);
}

Expression *eval_llvmceil(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, ceil(arg0->toReal()), type);
}

Expression *eval_llvmtrunc(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, trunc(arg0->toReal()), type);
}

Expression *eval_llvmround(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    return new RealExp(loc, round(arg0->toReal()), type);
}

Expression *eval_cttz(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);

    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t x = arg0->toInteger();

    int n = getBitsizeOfType(loc, type);

    if (x == 0)
    {
        if ((*arguments)[1]->toInteger())
            error(loc, "llvm.cttz.i#(0) is undefined");
    }
    else
    {
        int c = n >> 1;
        n -= 1;
        const uinteger_t mask = (static_cast<uinteger_t>(1L) << n) 
                                | (static_cast<uinteger_t>(1L) << n)-1;
        do {
            uinteger_t y = (x << c) & mask; if (y != 0) { n -= c; x = y; }
            c = c >> 1;
        } while (c != 0);
    }

    return new IntegerExp(loc, n, type);
}

Expression *eval_ctlz(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);

    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t x = arg0->toInteger();
    if (x == 0 && (*arguments)[1]->toInteger())
        error(loc, "llvm.ctlz.i#(0) is undefined");

    int n = getBitsizeOfType(loc, type);
    int c = n >> 1;
    do {
        uinteger_t y = x >> c; if (y != 0) { n -= c; x = y; }
        c = c >> 1;
    } while (c != 0);

    return new IntegerExp(loc, n - x, type);
}

Expression *eval_bswap(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Type* type = getTypeOfOverloadedIntrinsic(fd);

    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    #define BYTEMASK  0x00FF00FF00FF00FFLL
    #define SHORTMASK 0x0000FFFF0000FFFFLL
    #define INTMASK 0x00000000FFFFFFFFLL
#if WANT_CENT
    #define LONGMASK 0xFFFFFFFFFFFFFFFFLL
#endif
    switch (type->toBasetype()->ty)
    {
      case Tint128:
      case Tuns128:
#if WANT_CENT
          // swap high and low uints
          n = ((n >> 64) & LONGMASK) | ((n & LONGMASK) << 64);
#else
          error(loc, "cent/ucent not supported");
          break;
#endif
      case Tint64:
      case Tuns64:
          // swap high and low uints
          n = ((n >> 32) & INTMASK) | ((n & INTMASK) << 32);
      case Tint32:
      case Tuns32:
          // swap adjacent ushorts
          n = ((n >> 16) & SHORTMASK) | ((n & SHORTMASK) << 16);
      case Tint16:
      case Tuns16:
          // swap adjacent ubytes
          n = ((n >> 8 ) & BYTEMASK)  | ((n & BYTEMASK) << 8 );
          break;
      default:
          error(loc, "unsupported type");
          break;
    }
    return new IntegerExp(loc, n, type);
}

Expression *eval_ctpop(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    // FIXME Does not work for cent/ucent
    Type* type = getTypeOfOverloadedIntrinsic(fd);

    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    int cnt = 0;
    while (n)
    {
        cnt += (n & 1);
        n >>= 1;
    }
    return new IntegerExp(loc, cnt, type);
}
#else

Expression *eval_bsf(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    if (n == 0)
        error(loc, "bsf(0) is undefined");
    n = (n ^ (n - 1)) >> 1;  // convert trailing 0s to 1, and zero rest
    int k = 0;
    while( n )
    {   ++k;
        n >>=1;
    }
    return new IntegerExp(loc, k, Type::tint32);
}

Expression *eval_bsr(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    if (n == 0)
        error(loc, "bsr(0) is undefined");
    int k = 0;
    while(n >>= 1)
    {
        ++k;
    }
    return new IntegerExp(loc, k, Type::tint32);
}

Expression *eval_bswap(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    #define BYTEMASK  0x00FF00FF00FF00FFLL
    #define SHORTMASK 0x0000FFFF0000FFFFLL
    #define INTMASK 0x0000FFFF0000FFFFLL
    // swap adjacent ubytes
    n = ((n >> 8 ) & BYTEMASK)  | ((n & BYTEMASK) << 8 );
    // swap adjacent ushorts
    n = ((n >> 16) & SHORTMASK) | ((n & SHORTMASK) << 16);
    TY ty = arg0->type->toBasetype()->ty;
    // If 64 bits, we need to swap high and low uints
    if (ty == Tint64 || ty == Tuns64)
        n = ((n >> 32) & INTMASK) | ((n & INTMASK) << 32);
    return new IntegerExp(loc, n, arg0->type);
}
#endif

Expression *eval_popcnt(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKint64);
    uinteger_t n = arg0->toInteger();
    int cnt = 0;
    while (n)
    {
        cnt += (n & 1);
        n >>= 1;
    }
    return new IntegerExp(loc, cnt, arg0->type);
}

Expression *eval_yl2x(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    Expression *arg1 = (*arguments)[1];
    assert(arg1->op == TOKfloat64);
    longdouble x = arg0->toReal();
    longdouble y = arg1->toReal();
    longdouble result;
    Port::yl2x_impl(&x, &y, &result);
    return new RealExp(loc, result, arg0->type);
}

Expression *eval_yl2xp1(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    Expression *arg0 = (*arguments)[0];
    assert(arg0->op == TOKfloat64);
    Expression *arg1 = (*arguments)[1];
    assert(arg1->op == TOKfloat64);
    longdouble x = arg0->toReal();
    longdouble y = arg1->toReal();
    longdouble result;
    Port::yl2xp1_impl(&x, &y, &result);
    return new RealExp(loc, result, arg0->type);
}

void builtin_init()
{
#if IN_LLVM
    builtins._init(127); // Prime number like default value
#else
    builtins._init(47);
#endif

    // @safe @nogc pure nothrow real function(real)
    add_builtin("_D4core4math3sinFNaNbNiNfeZe", &eval_sin);
    add_builtin("_D4core4math3cosFNaNbNiNfeZe", &eval_cos);
    add_builtin("_D4core4math3tanFNaNbNiNfeZe", &eval_tan);
    add_builtin("_D4core4math4sqrtFNaNbNiNfeZe", &eval_sqrt);
    add_builtin("_D4core4math4fabsFNaNbNiNfeZe", &eval_fabs);
    add_builtin("_D4core4math5expm1FNaNbNiNfeZe", &eval_unimp);
    add_builtin("_D4core4math4exp21FNaNbNiNfeZe", &eval_unimp);

    // @trusted @nogc pure nothrow real function(real)
    add_builtin("_D4core4math3sinFNaNbNiNeeZe", &eval_sin);
    add_builtin("_D4core4math3cosFNaNbNiNeeZe", &eval_cos);
    add_builtin("_D4core4math3tanFNaNbNiNeeZe", &eval_tan);
    add_builtin("_D4core4math4sqrtFNaNbNiNeeZe", &eval_sqrt);
    add_builtin("_D4core4math4fabsFNaNbNiNeeZe", &eval_fabs);
    add_builtin("_D4core4math5expm1FNaNbNiNeeZe", &eval_unimp);
    add_builtin("_D4core4math4exp21FNaNbNiNeeZe", &eval_unimp);

    // @safe @nogc pure nothrow double function(double)
    add_builtin("_D4core4math4sqrtFNaNbNiNfdZd", &eval_sqrt);
    // @safe @nogc pure nothrow float function(float)
    add_builtin("_D4core4math4sqrtFNaNbNiNffZf", &eval_sqrt);

    // @safe @nogc pure nothrow real function(real, real)
    add_builtin("_D4core4math5atan2FNaNbNiNfeeZe", &eval_unimp);

    if (Port::yl2x_supported)
    {
        add_builtin("_D4core4math4yl2xFNaNbNiNfeeZe", &eval_yl2x);
    }
    else
    {
        add_builtin("_D4core4math4yl2xFNaNbNiNfeeZe", &eval_unimp);
    }

    if (Port::yl2xp1_supported)
    {
        add_builtin("_D4core4math6yl2xp1FNaNbNiNfeeZe", &eval_yl2xp1);
    }
    else
    {
        add_builtin("_D4core4math6yl2xp1FNaNbNiNfeeZe", &eval_unimp);
    }

    // @safe @nogc pure nothrow long function(real)
    add_builtin("_D4core4math6rndtolFNaNbNiNfeZl", &eval_unimp);

    // @safe @nogc pure nothrow real function(real)
    add_builtin("_D3std4math3sinFNaNbNiNfeZe", &eval_sin);
    add_builtin("_D3std4math3cosFNaNbNiNfeZe", &eval_cos);
    add_builtin("_D3std4math3tanFNaNbNiNfeZe", &eval_tan);
    add_builtin("_D3std4math4sqrtFNaNbNiNfeZe", &eval_sqrt);
    add_builtin("_D3std4math4fabsFNaNbNiNfeZe", &eval_fabs);
    add_builtin("_D3std4math5expm1FNaNbNiNfeZe", &eval_unimp);
    add_builtin("_D3std4math4exp21FNaNbNiNfeZe", &eval_unimp);

    // @trusted @nogc pure nothrow real function(real)
    add_builtin("_D3std4math3sinFNaNbNiNeeZe", &eval_sin);
    add_builtin("_D3std4math3cosFNaNbNiNeeZe", &eval_cos);
    add_builtin("_D3std4math3tanFNaNbNiNeeZe", &eval_tan);
    add_builtin("_D3std4math4sqrtFNaNbNiNeeZe", &eval_sqrt);
    add_builtin("_D3std4math4fabsFNaNbNiNeeZe", &eval_fabs);
    add_builtin("_D3std4math5expm1FNaNbNiNeeZe", &eval_unimp);
    add_builtin("_D3std4math4exp21FNaNbNiNeeZe", &eval_unimp);

    // @safe @nogc pure nothrow double function(double)
    add_builtin("_D3std4math4sqrtFNaNbNiNfdZd", &eval_sqrt);
    // @safe @nogc pure nothrow float function(float)
    add_builtin("_D3std4math4sqrtFNaNbNiNffZf", &eval_sqrt);

    // @safe @nogc pure nothrow real function(real, real)
    add_builtin("_D3std4math5atan2FNaNbNiNfeeZe", &eval_unimp);

    if (Port::yl2x_supported)
    {
        add_builtin("_D3std4math4yl2xFNaNbNiNfeeZe", &eval_yl2x);
    }
    else
    {
        add_builtin("_D3std4math4yl2xFNaNbNiNfeeZe", &eval_unimp);
    }

    if (Port::yl2xp1_supported)
    {
        add_builtin("_D3std4math6yl2xp1FNaNbNiNfeeZe", &eval_yl2xp1);
    }
    else
    {
        add_builtin("_D3std4math6yl2xp1FNaNbNiNfeeZe", &eval_unimp);
    }

    // @safe @nogc pure nothrow long function(real)
    add_builtin("_D3std4math6rndtolFNaNbNiNfeZl", &eval_unimp);

#if IN_LLVM
    // intrinsic llvm.sin.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.sin.f32", &eval_llvmsin);
    add_builtin("llvm.sin.f64", &eval_llvmsin);
    add_builtin("llvm.sin.f80", &eval_llvmsin);
    add_builtin("llvm.sin.f128", &eval_llvmsin);
    add_builtin("llvm.sin.ppcf128", &eval_llvmsin);

    // intrinsic llvm.cos.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.cos.f32", &eval_llvmcos);
    add_builtin("llvm.cos.f64", &eval_llvmcos);
    add_builtin("llvm.cos.f80", &eval_llvmcos);
    add_builtin("llvm.cos.f128", &eval_llvmcos);
    add_builtin("llvm.cos.ppcf128", &eval_llvmcos);

    // intrinsic llvm.sqrt.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.sqrt.f32", &eval_llvmsqrt);
    add_builtin("llvm.sqrt.f64", &eval_llvmsqrt);
    add_builtin("llvm.sqrt.f80", &eval_llvmsqrt);
    add_builtin("llvm.sqrt.f128", &eval_llvmsqrt);
    add_builtin("llvm.sqrt.ppcf128", &eval_llvmsqrt);

    // intrinsic llvm.fabs.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.fabs.f32", &eval_llvmfabs);
    add_builtin("llvm.fabs.f64", &eval_llvmfabs);
    add_builtin("llvm.fabs.f80", &eval_llvmfabs);
    add_builtin("llvm.fabs.f128", &eval_llvmfabs);
    add_builtin("llvm.fabs.ppcf128", &eval_llvmfabs);

    // intrinsic llvm.minnum.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.minnum.f32", &eval_llvmminnum);
    add_builtin("llvm.minnum.f64", &eval_llvmminnum);
    add_builtin("llvm.minnum.f80", &eval_llvmminnum);
    add_builtin("llvm.minnum.f128", &eval_llvmminnum);
    add_builtin("llvm.minnum.ppcf128", &eval_llvmminnum);

    // intrinsic llvm.maxnum.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.maxnum.f32", &eval_llvmmaxnum);
    add_builtin("llvm.maxnum.f64", &eval_llvmmaxnum);
    add_builtin("llvm.maxnum.f80", &eval_llvmmaxnum);
    add_builtin("llvm.maxnum.f128", &eval_llvmmaxnum);
    add_builtin("llvm.maxnum.ppcf128", &eval_llvmmaxnum);

    // intrinsic llvm.floor.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.floor.f32", &eval_llvmfloor);
    add_builtin("llvm.floor.f64", &eval_llvmfloor);
    add_builtin("llvm.floor.f80", &eval_llvmfloor);
    add_builtin("llvm.floor.f128", &eval_llvmfloor);
    add_builtin("llvm.floor.ppcf128", &eval_llvmfloor);

    // intrinsic llvm.ceil.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.ceil.f32", &eval_llvmceil);
    add_builtin("llvm.ceil.f64", &eval_llvmceil);
    add_builtin("llvm.ceil.f80", &eval_llvmceil);
    add_builtin("llvm.ceil.f128", &eval_llvmceil);
    add_builtin("llvm.ceil.ppcf128", &eval_llvmceil);

    // intrinsic llvm.trunc.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.trunc.f32", &eval_llvmtrunc);
    add_builtin("llvm.trunc.f64", &eval_llvmtrunc);
    add_builtin("llvm.trunc.f80", &eval_llvmtrunc);
    add_builtin("llvm.trunc.f128", &eval_llvmtrunc);
    add_builtin("llvm.trunc.ppcf128", &eval_llvmtrunc);

    // intrinsic llvm.round.f32/f64/f80/f128/ppcf128
    add_builtin("llvm.round.f32", &eval_llvmround);
    add_builtin("llvm.round.f64", &eval_llvmround);
    add_builtin("llvm.round.f80", &eval_llvmround);
    add_builtin("llvm.round.f128", &eval_llvmround);
    add_builtin("llvm.round.ppcf128", &eval_llvmround);

    // intrinsic llvm.bswap.i16/i32/i64/i128
    add_builtin("llvm.bswap.i16", &eval_bswap);
    add_builtin("llvm.bswap.i32", &eval_bswap);
    add_builtin("llvm.bswap.i64", &eval_bswap);
    add_builtin("llvm.bswap.i128", &eval_bswap);

    // intrinsic llvm.cttz.i8/i16/i32/i64/i128
    add_builtin("llvm.cttz.i8", &eval_cttz);
    add_builtin("llvm.cttz.i16", &eval_cttz);
    add_builtin("llvm.cttz.i32", &eval_cttz);
    add_builtin("llvm.cttz.i64", &eval_cttz);
    add_builtin("llvm.cttz.i128", &eval_cttz);

    // intrinsic llvm.ctlz.i8/i16/i32/i64/i128
    add_builtin("llvm.ctlz.i8", &eval_ctlz);
    add_builtin("llvm.ctlz.i16", &eval_ctlz);
    add_builtin("llvm.ctlz.i32", &eval_ctlz);
    add_builtin("llvm.ctlz.i64", &eval_ctlz);
    add_builtin("llvm.ctlz.i128", &eval_ctlz);

    // intrinsic llvm.ctpop.i8/i16/i32/i64/i128
    add_builtin("llvm.ctpop.i8", &eval_ctpop);
    add_builtin("llvm.ctpop.i16", &eval_ctpop);
    add_builtin("llvm.ctpop.i32", &eval_ctpop);
    add_builtin("llvm.ctpop.i64", &eval_ctpop);
    add_builtin("llvm.ctpop.i128", &eval_ctpop);
#else

    // @safe @nogc pure nothrow int function(uint)
    add_builtin("_D4core5bitop3bsfFNaNbNiNfkZi", &eval_bsf);
    add_builtin("_D4core5bitop3bsrFNaNbNiNfkZi", &eval_bsr);

    // @safe @nogc pure nothrow int function(ulong)
    add_builtin("_D4core5bitop3bsfFNaNbNiNfmZi", &eval_bsf);
    add_builtin("_D4core5bitop3bsrFNaNbNiNfmZi", &eval_bsr);

    // @safe @nogc pure nothrow uint function(uint)
    add_builtin("_D4core5bitop5bswapFNaNbNiNfkZk", &eval_bswap);

    // @safe @nogc pure nothrow int function(uint)
    add_builtin("_D4core5bitop7_popcntFNaNbNiNfkZi", &eval_popcnt);

    // @safe @nogc pure nothrow ushort function(ushort)
    add_builtin("_D4core5bitop7_popcntFNaNbNiNftZt", &eval_popcnt);

    // @safe @nogc pure nothrow int function(ulong)
    if (global.params.is64bit)
        add_builtin("_D4core5bitop7_popcntFNaNbNiNfmZi", &eval_popcnt);
#endif
}

/**********************************
 * Determine if function is a builtin one that we can
 * evaluate at compile time.
 */
BUILTIN isBuiltin(FuncDeclaration *fd)
{
    if (fd->builtin == BUILTINunknown)
    {
        builtin_fp fp = builtin_lookup(mangleExact(fd));
        fd->builtin = fp ? BUILTINyes : BUILTINno;
    }
    return fd->builtin;
}

/**************************************
 * Evaluate builtin function.
 * Return result; NULL if cannot evaluate it.
 */

Expression *eval_builtin(Loc loc, FuncDeclaration *fd, Expressions *arguments)
{
    if (fd->builtin == BUILTINyes)
    {
        builtin_fp fp = builtin_lookup(mangleExact(fd));
        assert(fp);
        return fp(loc, fd, arguments);
    }
    return NULL;
}
