// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-ll -O3 -of=%t.O3.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-s -of=%t.s %s
// RUN: %ldc -g -run %s
// RUN: %ldc -g -O3 -run %s

module mod;

import ldc.attributes;
import ldc.exception;

alias getCurrEx = getCurrentException;

class OurException(alias str) : Exception
{
    this() pure
    {
        super(str);
    }
}

class FooException : OurException!"Foo"
{
}

class BarException : OurException!"Bar"
{
}

class WoowException : OurException!"Woow"
{
}

class ZazzException : OurException!"Zazz"
{
}

void externalFunc() @weak
{
}

// CHECK-LABEL: define{{.*}} @{{.*}}test0
void test0()
{
    try
    {
        throw new FooException;
    }
    catch (FooException ex)
    {
        assert(ex is getCurrEx());

        externalFunc();
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}test1
void test1()
{
    try
    {
        throw new FooException();
    }
    catch (Exception)
    {
        auto ex = getCurrEx();
        assert(cast(FooException) ex);

        try
        {
            throw new BarException();
        }
        catch (BarException)
        {
            auto ex2 = getCurrEx();
            assert(cast(BarException) ex2);
        }

        assert(ex is getCurrEx()); // original exception restored
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}test2nested
void test2nested()
{
    try
    {
        try
        {
            throw new FooException();
        }
        catch (Exception)
        {
            auto ex = getCurrEx();
            assert(cast(FooException) ex);

            throw new BarException();
        }
    }
    catch (BarException)
    {
        auto ex = getCurrEx();
        assert(cast(BarException) ex);
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}test2
void test2()
{
    try
    {
        throw new BarException();
    }
    catch (BarException)
    {
        auto ex = getCurrEx();
        assert(cast(BarException) ex);

        test2nested();

        assert(ex is getCurrEx()); // original exception restored
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}test3
void test3()
{
    try
    {
        try
        {
            throw new BarException();
        }
        finally
        {
            auto ex = getCurrEx();
            assert(cast(BarException) ex);

            test2nested();

            assert(ex is getCurrEx()); // original exception restored
        }
    }
    catch (Exception)
    {
        auto ex = getCurrEx();
        assert(cast(BarException) ex);
    }

    assert(!getCurrEx());
}

struct DtorChecksBarException
{
    int* _i;
    this(int* i)
    {
        _i = i;
    }

    ~this()
    {
        auto ex = getCurrEx();
        assert(cast(BarException) ex);
        (*_i) *= 2;
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}test4
void test4()
{
    int i = 2;
    try
    {
        auto tmp = DtorChecksBarException(&i);
        throw new BarException();
    }
    catch (BarException)
    {
        auto ex = getCurrEx();
        assert(cast(BarException) ex);
        i += 7;
    }
    assert(i == 2 * 2 + 7);
}

// CHECK-LABEL: define{{.*}} @{{.*}}test5
void test5()
{
    try
    {
        throw new BarException();
    }
    catch (Exception)
    {
        debug externalFunc();
    }
    assert(!getCurrEx());
}

int test6()
{
    void throwsBar()
    {
        throw new BarException();
    }

    try
    {
        throwsBar();
        assert(0);
    }
    catch (FooException)
    {
        assert(0);
        return 1;
    }
    catch (Exception)
    {
        assert(cast(BarException) getCurrentException());
        return 2;
    }
    return 0;
}

struct DtorThrowsBarException
{
    ~this()
    {
        throw new BarException;
    }
}

/////////////////////////////////////////////////////////////

void testchain1_bar()
{
    assert(cast(ZazzException) getCurrEx());

    try
    {
        assert(cast(ZazzException) getCurrEx());

        scope (exit)
        {
            assert(cast(WoowException) getCurrEx());
            throw new BarException;
        }

        throw new WoowException();
    }
    catch (Exception e)
    {
        assert(cast(WoowException) e);
        assert(cast(BarException) e.next);

        assert(cast(WoowException) getCurrEx());
        assert(cast(BarException) getCurrEx().next);
    }

    assert(cast(ZazzException) getCurrEx());
}

void testchain1()
{
    assert(!getCurrEx());

    try
    {
        scope (exit)
        {
            assert(cast(ZazzException) getCurrEx());

            scope (exit)
                throw new FooException();

            testchain1_bar();

            assert(cast(ZazzException) getCurrEx());
        }

        throw new ZazzException();
    }
    catch (Exception e)
    {
        assert(cast(ZazzException) e);
        assert(cast(FooException) e.next);

        assert(cast(ZazzException) getCurrEx());
        assert(cast(FooException) getCurrEx().next);
    }
}

/////////////////////////////////////////////////////////////

void testrethrow_throws()
{
    assert(!getCurrEx());

    try
    {
        throw new ZazzException();
    }
    catch (Exception e)
    {
        assert(cast(ZazzException) e);
        assert(cast(ZazzException) getCurrEx());

        throw e;
    }

    assert(cast(ZazzException) getCurrEx());
}

void testrethrow()
{
    assert(!getCurrEx());

    try
    {
        testrethrow_throws();
    }
    catch (Exception)
    {
        assert(cast(ZazzException) getCurrEx());
    }

    assert(!getCurrEx());
}

/////////////////////////////////////////////////////////////

void testexample()
{
    void checkException(FooException ex)
    {
        assert(ex is getCurrEx());
    }

    try
    {
        checkException(null);

        auto ex = new FooException;
        scope (exit)
            checkException(ex);

        throw ex;
    }
    catch (FooException ex)
    {
        checkException(ex);
    }
}

/////////////////////////////////////////////////////////////

void main()
{
    assert(!getCurrEx());

    test0();
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();

    testchain1();

    testrethrow();

    testexample();

    assert(!getCurrEx());
}
