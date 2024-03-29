import dshell;

int main()
{
    version (LDC) return DISABLED; // requires shared libs and a patched cmdline below; hardly worth it

    if (OS != "linux")
    {
        writefln("Skipping shared library test on %s.", OS);
        return DISABLED;
    }

    run("$DMD -m$MODEL -of$OUTPUT_BASE/a$EXE -defaultlib=libphobos2.so $EXTRA_FILES/test_shared.d");
    run("$OUTPUT_BASE/a$EXE", stdout, stderr, [
        "LD_LIBRARY_PATH" : "../../../phobos/generated/"~OS~'/'~BUILD~'/'~MODEL
    ]);

    return 0;
}
