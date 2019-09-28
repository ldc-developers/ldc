/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/gluelayer.d, _gluelayer.d)
 * Documentation:  https://dlang.org/phobos/dmd_gluelayer.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/gluelayer.d
 */

module dmd.gluelayer;

import dmd.dmodule;
import dmd.dscope;
import dmd.dsymbol;
import dmd.mtype;
import dmd.statement;
import dmd.root.file;

version (IN_LLVM)
{
    struct Symbol;
    struct code;
    struct block;
    struct Blockx;
    struct elem;
    struct TYPE;
    alias type = TYPE;

    extern (C++)
    {
        Statement asmSemantic(AsmStatement s, Scope* sc);
        void objc_initSymbols() {}
    }
}
else version (NoBackend)
{
    import dmd.lib : Library;

    struct Symbol;
    struct code;
    struct block;
    struct Blockx;
    struct elem;
    struct TYPE;
    alias type = TYPE;

    extern (C++)
    {
        // glue
        void obj_write_deferred(Library library)        {}
        void obj_start(const(char)* srcfile)            {}
        void obj_end(Library library, const(char)* objfilename) {}
        void genObjFile(Module m, bool multiobj)        {}

        // msc
        void backend_init() {}
        void backend_term() {}

        // iasm
        Statement asmSemantic(AsmStatement s, Scope* sc)
        {
            sc.func.hasReturnExp = 8;
            return null;
        }

        // toir
        void toObjFile(Dsymbol ds, bool multiobj)   {}

        extern(C++) abstract class ObjcGlue
        {
            static void initialize() {}
        }
    }
}
else version (MARS)
{
    import dmd.lib : Library;

    public import dmd.backend.cc : block, Blockx, Symbol;
    public import dmd.backend.type : type;
    public import dmd.backend.el : elem;
    public import dmd.backend.code_x86 : code;

    extern (C++)
    {
        void obj_write_deferred(Library library);
        void obj_start(const(char)* srcfile);
        void obj_end(Library library, const(char)* objfilename);
        void genObjFile(Module m, bool multiobj);

        void backend_init();
        void backend_term();

        Statement asmSemantic(AsmStatement s, Scope* sc);

        void toObjFile(Dsymbol ds, bool multiobj);

        extern(C++) abstract class ObjcGlue
        {
            static void initialize();
        }
    }
}
else version (IN_GCC)
{
    extern (C++) union tree_node;

    alias Symbol = tree_node;
    alias code = tree_node;
    alias type = tree_node;

    extern (C++)
    {
        Statement asmSemantic(AsmStatement s, Scope* sc);
    }

    // stubs
    extern(C++) abstract class ObjcGlue
    {
        static void initialize() {}
    }
}
else
    static assert(false, "Unsupported compiler backend");
