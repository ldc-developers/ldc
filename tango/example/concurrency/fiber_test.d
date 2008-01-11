import tango.core.Thread;

extern (C) int printf(char * str, ...);

void main()
{
	printf("Compile with -unittest");
}


unittest
{
    printf("Testing context creation/deletion\n");
    int s0 = 0;
    static int s1 = 0;
    
    Fiber a = new Fiber(
    delegate void()
    {
        s0++;
    });
    
    static void fb() { s1++; }
    
    Fiber b = new Fiber(&fb);
    
    Fiber c = new Fiber(
        delegate void() { assert(false); });
    
    assert(a);
    assert(b);
    assert(c);
    
    assert(s0 == 0);
    assert(s1 == 0);
    assert(a.state == Fiber.State.HOLD);
    assert(b.state == Fiber.State.HOLD);
    assert(c.state == Fiber.State.HOLD);
    
    delete c;
    
    assert(s0 == 0);
    assert(s1 == 0);
    assert(a.state == Fiber.State.HOLD);
    assert(b.state == Fiber.State.HOLD);
    
    printf("running a\n");
    a.call();
    printf("done a\n");
    
    assert(a);
    
    assert(s0 == 1);
    assert(s1 == 0);
    assert(a.state == Fiber.State.TERM);
    assert(b.state == Fiber.State.HOLD);    
    
    assert(b.state == Fiber.State.HOLD);
    
    printf("Running b\n");
    b.call();
    printf("Done b\n");
    
    assert(s0 == 1);
    assert(s1 == 1);
    assert(b.state == Fiber.State.TERM);
    
    delete a;
    delete b;
    
    printf("Context creation passed\n");
}
    
unittest
{
    printf("Testing context switching\n");
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    
    Fiber a = new Fiber(
    delegate void()
    {
        while(true)
        {
            debug printf(" ---A---\n");
            s0++;
            Fiber.yield();
        }
    });
    
    
    Fiber b = new Fiber(
    delegate void()
    {
        while(true)
        {
            debug printf(" ---B---\n");
            s1++;
            Fiber.yield();
        }
    });
    
    
    Fiber c = new Fiber(
    delegate void()
    {
        while(true)
        {
            debug printf(" ---C---\n");
            s2++;
            Fiber.yield();
        }
    });
    
    assert(a);
    assert(b);
    assert(c);
    assert(s0 == 0);
    assert(s1 == 0);
    assert(s2 == 0);
    
    a.call();
    b.call();
    
    assert(a);
    assert(b);
    assert(c);
    assert(s0 == 1);
    assert(s1 == 1);
    assert(s2 == 0);
    
    for(int i=0; i<20; i++)
    {
        c.call();
        a.call();
    }
    
    assert(a);
    assert(b);
    assert(c);
    assert(s0 == 21);
    assert(s1 == 1);
    assert(s2 == 20);
    
    delete a;
    delete b;
    delete c;
    
    printf("Context switching passed\n");
}
    
unittest
{
    printf("Testing nested contexts\n");
    Fiber a, b, c;
    
    int t0 = 0;
    int t1 = 0;
    int t2 = 0;
    
    a = new Fiber(
    delegate void()
    {
        
        t0++;
        b.call();
        
    });
    
    b = new Fiber(
    delegate void()
    {
        assert(t0 == 1);
        assert(t1 == 0);
        assert(t2 == 0);
        
        t1++;
        c.call();
        
    });
    
    c = new Fiber(
    delegate void()
    {
        assert(t0 == 1);
        assert(t1 == 1);
        assert(t2 == 0);
        
        t2++;
    });
    
    assert(a);
    assert(b);
    assert(c);
    assert(t0 == 0);
    assert(t1 == 0);
    assert(t2 == 0);
    
    a.call();
    
    assert(t0 == 1);
    assert(t1 == 1);
    assert(t2 == 1);
    
    assert(a);
    assert(b);
    assert(c);
    
    delete a;
    delete b;
    delete c;
    
    printf("Nesting contexts passed\n");
}

unittest
{
	printf("Testing basic exceptions\n");


	int t0 = 0;
	int t1 = 0;
	int t2 = 0;

	assert(t0 == 0);
	assert(t1 == 0);
	assert(t2 == 0);

	try
	{

		try
		{
			throw new Exception("Testing\n");
			t2++;
		}
		catch(Exception fx)
		{
			t1++;
			throw fx;
		}
	
		t2++;
	}
	catch(Exception ex)
	{
		t0++;
		printf("%.*s\n", ex.toString);
	}

	assert(t0 == 1);
	assert(t1 == 1);
	assert(t2 == 0);

	printf("Basic exceptions are supported\n");
}


unittest
{
    printf("Testing exceptions\n");
    Fiber a, b, c;
    
    int t0 = 0;
    int t1 = 0;
    int t2 = 0;
    
    printf("t0 = %d\nt1 = %d\nt2 = %d\n", t0, t1, t2);
    
    a = new Fiber(
    delegate void()
    {
        t0++;
        throw new Exception("A exception\n");
        t0++;
    });
    
    b = new Fiber(
    delegate void()
    {
        t1++;
        c.call();
        t1++;
    });
    
    c = new Fiber(
    delegate void()
    {
        t2++;
        throw new Exception("C exception\n");
        t2++;
    });
    
    assert(a);
    assert(b);
    assert(c);
    assert(t0 == 0);
    assert(t1 == 0);
    assert(t2 == 0);
    
    try
    {
        a.call();
        assert(false);
    }
    catch(Exception e)
    {
        printf("%.*s\n", e.toString);
    }
    
    assert(a);
    assert(a.state == Fiber.State.TERM);
    assert(b);
    assert(c);
    assert(t0 == 1);
    assert(t1 == 0);
    assert(t2 == 0);
    
    try
    {
        b.call();
        assert(false);
    }
    catch(Exception e)
    {
        printf("%.*s\n", e.toString);
    }
    
    printf("blah2\n");
    
    assert(a);
    assert(b);
    assert(b.state == Fiber.State.TERM);
    assert(c);
    assert(c.state == Fiber.State.TERM);
    assert(t0 == 1);
    assert(t1 == 1);
    assert(t2 == 1);

	delete a;
	delete b;
	delete c;
    

	Fiber t;
	int q0 = 0;
	int q1 = 0;

	t = new Fiber(
	delegate void()
	{
		try
		{
			q0++;
			throw new Exception("T exception\n");
			q0++;
		}
		catch(Exception ex)
		{
			q1++;
			printf("!!!!!!!!GOT EXCEPTION!!!!!!!!\n");
			printf("%.*s\n", ex.toString);
		}
	});


	assert(t);
	assert(q0 == 0);
	assert(q1 == 0);
	t.call();
	assert(t);
	assert(t.state == Fiber.State.TERM);
	assert(q0 == 1);
	assert(q1 == 1);

	delete t;
   
    Fiber d, e;
    int s0 = 0;
    int s1 = 0;
    
    d = new Fiber(
    delegate void()
    {
        try
        {
            s0++;
            e.call();
            Fiber.yield();
            s0++;
            e.call();
            s0++;
        }
        catch(Exception ex)
        {
            printf("%.*s\n", ex.toString);
        }
    });
    
    e = new Fiber(
    delegate void()
    {
        s1++;
        Fiber.yield();
        throw new Exception("E exception\n");
        s1++;
    });
    
    assert(d);
    assert(e);
    assert(s0 == 0);
    assert(s1 == 0);
    
    d.call();
    
    assert(d);
    assert(e);
    assert(s0 == 1);
    assert(s1 == 1);
    
    d.call();
    
    assert(d);
    assert(e);
    assert(s0 == 2);
    assert(s1 == 1);
    
    assert(d.state == Fiber.State.TERM);
    assert(e.state == Fiber.State.TERM);
    
    delete d;
    delete e;
    
    printf("Exceptions passed\n");
}

unittest
{
    printf("Testing standard exceptions\n");
    int t = 0;
    
    Fiber a = new Fiber(
    delegate void()
    {
        throw new Exception("BLAHAHA");
    });
    
    assert(a);
    assert(t == 0);
    
    try
    {
        a.call();
        assert(false);
    }
    catch(Exception e)
    {
        printf("%.*s\n", e.toString);
    }
    
    assert(a);
    assert(a.state == Fiber.State.TERM);
    assert(t == 0);
    
    delete a;
    
    
    printf("Standard exceptions passed\n");
}

unittest
{
    printf("Memory stress test\n");
    
    const uint STRESS_SIZE = 5000;
    
    Fiber ctx[];
    ctx.length = STRESS_SIZE;
    
    int cnt0 = 0;
    int cnt1 = 0;
    
    void threadFunc()
    {
        cnt0++;
        Fiber.yield;
        cnt1++;
    }
    
    foreach(inout Fiber c; ctx)
    {
        c = new Fiber(&threadFunc, 1024);
    }
    
    assert(cnt0 == 0);
    assert(cnt1 == 0);
    
    foreach(inout Fiber c; ctx)
    {
        c.call;
    }
    
    assert(cnt0 == STRESS_SIZE);
    assert(cnt1 == 0);
    
    foreach(inout Fiber c; ctx)
    {
        c.call;
    }
    
    assert(cnt0 == STRESS_SIZE);
    assert(cnt1 == STRESS_SIZE);
    
    foreach(inout Fiber c; ctx)
    {
        delete c;
    }
    
    assert(cnt0 == STRESS_SIZE);
    assert(cnt1 == STRESS_SIZE);
    
    printf("Memory stress test passed\n");
}

unittest
{
    printf("Testing floating point\n");
    
    float f0 = 1.0;
    float f1 = 0.0;
    
    double d0 = 2.0;
    double d1 = 0.0;
    
    real r0 = 3.0;
    real r1 = 0.0;
    
    assert(f0 == 1.0);
    assert(f1 == 0.0);
    assert(d0 == 2.0);
    assert(d1 == 0.0);
    assert(r0 == 3.0);
    assert(r1 == 0.0);
    
    Fiber a, b, c;
    
    a = new Fiber(
    delegate void()
    {
        while(true)
        {
            f0 ++;
            d0 ++;
            r0 ++;
            
            Fiber.yield();
        }
    });
    
    b = new Fiber(
    delegate void()
    {
        while(true)
        {
            f1 = d0 + r0;
            d1 = f0 + r0;
            r1 = f0 + d0;
            
            Fiber.yield();
        }
    });
    
    c = new Fiber(
    delegate void()
    {
        while(true)
        {
            f0 *= d1;
            d0 *= r1;
            r0 *= f1;
            
            Fiber.yield();
        }
    });
    
    a.call();
    assert(f0 == 2.0);
    assert(f1 == 0.0);
    assert(d0 == 3.0);
    assert(d1 == 0.0);
    assert(r0 == 4.0);
    assert(r1 == 0.0);
    
    b.call();
    assert(f0 == 2.0);
    assert(f1 == 7.0);
    assert(d0 == 3.0);
    assert(d1 == 6.0);
    assert(r0 == 4.0);
    assert(r1 == 5.0);
    
    c.call();
    assert(f0 == 12.0);
    assert(f1 == 7.0);
    assert(d0 == 15.0);
    assert(d1 == 6.0);
    assert(r0 == 28.0);
    assert(r1 == 5.0);
    
    a.call();
    assert(f0 == 13.0);
    assert(f1 == 7.0);
    assert(d0 == 16.0);
    assert(d1 == 6.0);
    assert(r0 == 29.0);
    assert(r1 == 5.0);
    
    printf("Floating point passed\n");
}


version(x86) unittest
{
    printf("Testing registers\n");
    
    struct registers
    {
        int eax, ebx, ecx, edx;
        int esi, edi;
        int ebp, esp;
        
        //TODO: Add fpu stuff
    }
    
    static registers old;
    static registers next;
    static registers g_old;
    static registers g_next;
    
    //I believe that D calling convention requires that
    //EBX, ESI and EDI be saved.  In order to validate
    //this, we write to those registers and call the
    //stack thread.
    static StackThread reg_test = new StackThread(
    delegate void() 
    {
        asm
        {
            naked;
            
            pushad;
            
            mov EBX, 1;
            mov ESI, 2;
            mov EDI, 3;
            
            mov [old.ebx], EBX;
            mov [old.esi], ESI;
            mov [old.edi], EDI;
            mov [old.ebp], EBP;
            mov [old.esp], ESP;
            
            call StackThread.yield;
            
            mov [next.ebx], EBX;
            mov [next.esi], ESI;
            mov [next.edi], EDI;
            mov [next.ebp], EBP;
            mov [next.esp], ESP;
            
            popad;
        }
    });
    
    //Run the stack context
    asm
    {
        naked;
        
        pushad;
        
        mov EBX, 10;
        mov ESI, 11;
        mov EDI, 12;
        
        mov [g_old.ebx], EBX;
        mov [g_old.esi], ESI;
        mov [g_old.edi], EDI;
        mov [g_old.ebp], EBP;
        mov [g_old.esp], ESP;
        
        mov EAX, [reg_test];
        call StackThread.call;
        
        mov [g_next.ebx], EBX;
        mov [g_next.esi], ESI;
        mov [g_next.edi], EDI;
        mov [g_next.ebp], EBP;
        mov [g_next.esp], ESP;
        
        popad;
    }
    
    
    //Make sure the registers are byte for byte equal.
    assert(old.ebx = 1);
    assert(old.esi = 2);
    assert(old.edi = 3);
    assert(old == next);
    
    assert(g_old.ebx = 10);
    assert(g_old.esi = 11);
    assert(g_old.edi = 12);
    assert(g_old == g_next);
    
    printf("Registers passed!\n");
}


unittest
{
    printf("Testing throwYield\n");
    
    int q0 = 0;
    
    Fiber st0 = new Fiber(
    delegate void()
    {
        q0++;
        Fiber.yieldAndThrow(new Exception("testing throw yield\n"));
        q0++;
    });
    
    try
    {
        st0.call();
        assert(false);
    }
    catch(Exception e)
    {
        printf("%.*s\n", e.toString);
    }
    
    assert(q0 == 1);
    assert(st0.state == Fiber.State.HOLD);
    
    st0.call();
    assert(q0 == 2);
    assert(st0.state == Fiber.State.TERM);
    
    printf("throwYield passed!\n");
}

unittest
{
    printf("Testing thread safety\n");
    
    int x = 0, y = 0;
    
    Fiber sc0 = new Fiber(
    {
        while(true)
        {
            x++;
            Fiber.yield;
        }
    });
    
    Fiber sc1 = new Fiber(
    {
        while(true)
        {
            y++;
            Fiber.yield;
        }
    });
    
    Thread t0 = new Thread(
    {
        for(int i=0; i<10000; i++)
            sc0.call();
    });
    
    Thread t1 = new Thread(
    {
        for(int i=0; i<10000; i++)
            sc1.call();
    });
    
    assert(sc0);
    assert(sc1);
    assert(t0);
    assert(t1);
    
    t0.start;
    t1.start;
    t0.join;
    t1.join;
    
    assert(x == 10000);
    assert(y == 10000);
    
    printf("Thread safety passed!\n");
}

