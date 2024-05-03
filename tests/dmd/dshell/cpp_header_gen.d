module test.dshell.cpp_header_gen;

import dshell;

int main()
{
    if (!CXX.length)
    {
        writeln("CPP header generation test was skipped because $CXX is empty!");
        return DISABLED;
    }
    // DMC cannot compile the generated headers ...
    version (Windows)
    {
        import std.algorithm : canFind;
        if (CXX.canFind("dmc"))
        {
            writeln("CPP header generation test was skipped because DMC is not supported!");
            return DISABLED;
        }
    }

    Vars.set("SOURCE_DIR",  "$EXTRA_FILES/cpp_header_gen");
    Vars.set("LIB",         "$OUTPUT_BASE/library$LIBEXT");
    Vars.set("CPP_OBJ",     "$OUTPUT_BASE/cpp$OBJ");
    Vars.set("HEADER_EXE",  "$OUTPUT_BASE/test$EXE");

    run("$DMD -m$MODEL -c -lib -of=$LIB -HC=verbose -HCf=$OUTPUT_BASE/library.h $SOURCE_DIR/library.d");

    // Dump header if any of the following step fails
    scope (failure)
    {
        const file = buildPath(Vars.OUTPUT_BASE, "library.h");
        const header = (cast(string) read(file)).ifThrown("<Could not read file>\n");

        stderr.flush();
        writeln("========================= library.h ==================================\n");
        write(header);
        writeln("======================================================================\n");
        stdout.flush();
    }

    // LDC: don't specify -m<model> for C compiler for non-x86 targets, it's mostly unsupported.
    version (X86)         enum X86_Any = true;
    else version (X86_64) enum X86_Any = true;
    else                  enum X86_Any = false;

    version (Windows)
        run([CXX, "/c", "/Fo" ~ Vars.CPP_OBJ, "/I" ~ OUTPUT_BASE, "/I" ~ EXTRA_FILES ~"/../../../../../dmd/root", Vars.SOURCE_DIR ~ "/app.cpp"]);
    else static if (X86_Any)
        run("$CXX -std=c++11 -m$MODEL -c -o $CPP_OBJ -I$OUTPUT_BASE -I$EXTRA_FILES/../../../../../dmd/root $SOURCE_DIR/app.cpp");
    else
        run("$CXX -std=c++11 -c -o $CPP_OBJ -I$OUTPUT_BASE -I$EXTRA_FILES/../../../../../dmd/root $SOURCE_DIR/app.cpp");
    run("$DMD -m$MODEL -of=$HEADER_EXE $LIB $CPP_OBJ");
    run("$HEADER_EXE");

    return 0;
}
