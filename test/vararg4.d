module vararg4;
import std.stdarg;

void vafunc(...)
{
    foreach(i,v; _arguments) {
        if (typeid(byte) == v) {
            printf("byte(%d)\n", va_arg!(byte)(_argptr));
        }
        else if (typeid(short) == v) {
            printf("short(%d)\n", va_arg!(short)(_argptr));
        }
        else if (typeid(int) == v) {
            printf("int(%d)\n", va_arg!(int)(_argptr));
        }
        else if (typeid(long) == v) {
            printf("long(%ld)\n", va_arg!(long)(_argptr));
        }
        else if (typeid(float) == v) {
            printf("float(%f)\n", va_arg!(float)(_argptr));
        }
        else if (typeid(double) == v) {
            printf("double(%f)\n", va_arg!(double)(_argptr));
        }
        else if (typeid(real) == v) {
            printf("real(%f)\n", va_arg!(real)(_argptr));
        }
        else
        assert(0, "unsupported type");
    }
}

void main()
{
    vafunc(byte.max,short.max,1,2,3,4L,5.0f,6.0,cast(real)7);
}
