import core.runtime;
import core.atomic;
import core.stdc.string;
import core.sys.posix.dlfcn;

shared uint tlsDtor, dtor;
void staticDtorHook() { atomicOp!"+="(tlsDtor, 1); }
void sharedStaticDtorHook() { atomicOp!"+="(dtor, 1); }

version (LDC) version (darwin) version = LDC_darwin;

void runTest(string name)
{
    auto h = Runtime.loadLibrary(name);
    assert(h !is null);

    *cast(void function()*).dlsym(h, "_D9lib_1341414staticDtorHookOPFZv") = &staticDtorHook;
    *cast(void function()*).dlsym(h, "_D9lib_1341420sharedStaticDtorHookOPFZv") = &sharedStaticDtorHook;

    const unloaded = Runtime.unloadLibrary(h);
    version (CRuntime_Musl)
    {
        // On Musl, unloadLibrary is a no-op because dlclose is a no-op
        assert(!unloaded);
        assert(tlsDtor == 0);
        assert(dtor == 0);
    }
    else
    {
        assert(unloaded);
        assert(tlsDtor == 1);
        version (LDC_darwin)
        {
            // Since 10.13: https://github.com/ldc-developers/ldc/issues/3002
            assert(dtor == 0);
        }
        else
            assert(dtor == 1);
    }
}

void main(string[] args)
{
    auto name = args[0] ~ '\0';
    const pathlen = strrchr(name.ptr, '/') - name.ptr + 1;
    name = name[0 .. pathlen] ~ "lib_13414.so";

    runTest(name);
}
