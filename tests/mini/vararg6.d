// tries to implement a fairly complete variadic print function
module tangotests.vararg3;

extern(C) int printf(char*, ...);

struct User
{
    char[] name;
    char[] nick;
    uint age;

    char[] toString()
    {
        return nick ~ "(" ~ name ~ ")";
    }
}

struct Infidel
{
    char[] whocares;
}

class Obj
{
    private char[] ty;

    this(char[] t)
    {
        ty = t;
    }

    char[] toString()
    {
        return "Obj(" ~ ty ~ ")";
    }
}

 struct TLA
 {
    char[3] acronym;

    char[] toString()
    {
        return acronym;
    }
}

void main()
{
    User user = User("Bob Doe", "bd", 47);
    char[] str = user.toString();
    printf("Direct call:\n%.*s\nBy typeinfo:\n", str.length, str.ptr);
    print(user, '\n');

    print("Without toString:\n");
    Infidel inf = Infidel("not me");
    print(inf, '\n');

    print("Character arrays:\n");
    print("hello world\n");

    print("Signed integers:\n");
    print(cast(byte)byte.max,' ',cast(short)short.max,' ',cast(int)int.max,' ',cast(long)long.max,'\n');

    print("Unsigned integers:\n");
    print(cast(ubyte)ubyte.max,' ',cast(ushort)ushort.max,' ',cast(uint)uint.max,' ',cast(ulong)ulong.max,'\n');

    print("Floating point:\n");
    print(cast(float)1.273f, ' ', cast(double)3.412367, ' ', cast(real)142.96731112, '\n');

    print("Arrays:\n");
    int[] ia1 = [1,2,3,4,5,6,7,8,9];
    print(ia1, '\n');
    float[] fa1 = [0.1f, 0.15f, 0.2f, 0.25f, 0.3f];
    print(fa1, '\n');

    print("Pointers:\n");
    print(&user,'\n');
    print(&inf,'\n');
    print(&ia1,'\n');
    print(&fa1,'\n');

    print("Static arrays:\n");
    int[5] isa1 = [1,2,4,8,16];
    print(isa1,'\n');

    print("Classes:\n");
    Obj o = new Obj("foo");
    print(o, '\n');

    print("Mixed:\n");
    print(123, ' ', 42.536f, " foobar ", ia1, ' ', user, '\n');
    print(42, ' ', cast(byte)12, ' ', user, ' ', cast(short)1445, " foo\n");

    print("International:\n");
    print('æ','ø','å','\n');
    print('Æ','Ø','Å','\n');
    print("rød grød med fløde\n");
    print("Heiße\n");

    print("TLAs:\n");
    TLA tla1 = TLA("FBI");
    TLA tla2 = TLA("CIA");
    TLA tla3 = TLA("TLA");
    print(tla1);
    print(tla2);
    print(tla3, '\n');
    print(tla1, tla2, tla3, '\n');
    print(TLA("FBI"), TLA("CIA"), TLA("TLA"), '\n');

    print("Done!\n");
}

private void* get_va_arg(TypeInfo ti, ref void* vp)
{
    void* arg = vp;
    vp = vp + ( ( ti.tsize + size_t.sizeof - 1 ) & ~( size_t.sizeof - 1 ) );
    return arg;
}

void print(TypeInfo ti, void* arg)
{
    if (ti == typeid(byte))
        printf("%d", *cast(byte*)arg);
    else if (ti == typeid(short))
        printf("%d", *cast(short*)arg);
    else if (ti == typeid(int))
        printf("%d", *cast(int*)arg);
    else if (ti == typeid(long))
        printf("%lld", *cast(long*)arg);

    else if (ti == typeid(ubyte))
        printf("%u", *cast(ubyte*)arg);
    else if (ti == typeid(ushort))
        printf("%u", *cast(ushort*)arg);
    else if (ti == typeid(uint))
        printf("%u", *cast(uint*)arg);
    else if (ti == typeid(ulong))
        printf("%llu", *cast(ulong*)arg);

    else if (ti == typeid(float))
        printf("%f", *cast(float*)arg);
    else if (ti == typeid(double))
        printf("%f", *cast(double*)arg);
    else if (ti == typeid(real)) // FIXME: 80bit?
    {
        version(LLVM_X86_FP80)
            printf("%llf", *cast(real*)arg);
        else
            printf("%f", *cast(real*)arg);
    }

    else if (ti == typeid(char))
        printf("%.*s", 1, arg);
    else if (ti == typeid(wchar))
        printf("%.*s", 2, arg);
    else if (ti == typeid(dchar))
        printf("%.*s", 4, arg);

    else if (ti == typeid(char[]))
    {
        char[] str = *cast(char[]*)arg;
        printf("%.*s", str.length, str.ptr);
    }
    else if (ti == typeid(wchar[]))
    {
        wchar[] str = *cast(wchar[]*)arg;
        printf("%.*s", str.length*2, str.ptr);
    }
    else if (ti == typeid(dchar[]))
    {
        dchar[] str = *cast(dchar[]*)arg;
        printf("%.*s", str.length*4, str.ptr);
    }

    else if (auto pti = cast(TypeInfo_Pointer)ti)
    {
        printf("%p", *cast(void**)arg);
    }

    else if (auto sti = cast(TypeInfo_Struct)ti)
    {
        if (sti.xtoString !is null)
        {
            char[] str = sti.xtoString(arg);
            printf("%.*s", str.length, str.ptr);
        }
        else
        {
            char[] str = sti.toString();
            printf("%.*s", str.length, str.ptr);
        }
    }

    else if (auto ati = cast(TypeInfo_Array)ti)
    {
        auto tnext = ati.next;
        size_t len = *cast(size_t*)arg;
        void* ptr = *(cast(void**)arg + 1);
        printf("[");
        for(auto i=0; i<len; ++i)
        {
            print(tnext, get_va_arg(tnext, ptr));
            if (i < len-1)
                printf(",");
        }
        printf("]");
    }

    else if (auto cti = cast(TypeInfo_Class)ti)
    {
        auto o = *cast(Object*)arg;
        char[] str = o.toString();
        printf("%.*s", str.length, str.ptr);
    }

    // static arrays are converted to dynamic arrays when passed to variadic functions
    else if (auto sati = cast(TypeInfo_StaticArray)ti)
    {
        assert(0, "static arrays not supported");
    }

    else if (auto aati = cast(TypeInfo_AssociativeArray)ti)
    {
        assert(0, "associative array not supported");
    }

    else
    {
        char[] str = ti.toString();
        printf("typeinfo: %.*s\n", str.length, str.ptr);
        str = ti.classinfo.name;
        printf("typeinfo.classinfo: %.*s\n", str.length, str.ptr);
        assert(0, "unsupported type ^");
    }
}

void print(...)
{
    void* argptr = _argptr;
    assert(argptr);

    foreach(i,ti; _arguments)
    {
        void* arg = get_va_arg(ti, argptr);
        assert(arg);
        print(ti, arg);
    }
}
