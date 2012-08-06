
// Compiler implementation of the D programming language
// Copyright (c) 1999-2011 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// http://www.dsource.org/projects/dmd/browser/trunk/src/idgen.c
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

// Program to generate string files in d data structures.
// Saves much tedious typing, and eliminates typo problems.
// Generates:
//      id.h
//      id.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct Msgtable
{
        const char *ident;      // name to use in DMD source
        const char *name;       // name in D executable
};

Msgtable msgtable[] =
{
    { "IUnknown" },
    { "Object" },
    { "object" },
    { "max" },
    { "min" },
    { "This", "this" },
    { "super" },
    { "ctor", "__ctor" },
    { "dtor", "__dtor" },
    { "cpctor", "__cpctor" },
    { "_postblit", "__postblit" },
    { "classInvariant", "__invariant" },
    { "unitTest", "__unitTest" },
    { "require", "__require" },
    { "ensure", "__ensure" },
    { "init" },
    { "size" },
    { "__sizeof", "sizeof" },
    { "__xalignof", "alignof" },
    { "mangleof" },
    { "stringof" },
    { "tupleof" },
    { "length" },
    { "remove" },
    { "ptr" },
    { "array" },
    { "funcptr" },
    { "dollar", "__dollar" },
    { "ctfe", "__ctfe" },
    { "offset" },
    { "offsetof" },
    { "ModuleInfo" },
    { "ClassInfo" },
    { "classinfo" },
    { "typeinfo" },
    { "outer" },
    { "Exception" },
    { "AssociativeArray" },
    { "Throwable" },
    { "Error" },
    { "withSym", "__withSym" },
    { "result", "__result" },
    { "returnLabel", "__returnLabel" },
    { "delegate" },
    { "line" },
    { "empty", "" },
    { "p" },
    { "q" },
    { "coverage", "__coverage" },
    { "__vptr" },
    { "__monitor" },

    { "TypeInfo" },
    { "TypeInfo_Class" },
    { "TypeInfo_Interface" },
    { "TypeInfo_Struct" },
    { "TypeInfo_Enum" },
    { "TypeInfo_Typedef" },
    { "TypeInfo_Pointer" },
    { "TypeInfo_Vector" },
    { "TypeInfo_Array" },
    { "TypeInfo_StaticArray" },
    { "TypeInfo_AssociativeArray" },
    { "TypeInfo_Function" },
    { "TypeInfo_Delegate" },
    { "TypeInfo_Tuple" },
    { "TypeInfo_Const" },
    { "TypeInfo_Invariant" },
    { "TypeInfo_Shared" },
    { "TypeInfo_Wild", "TypeInfo_Inout" },
    { "elements" },
    { "_arguments_typeinfo" },
    { "_arguments" },
    { "_argptr" },
    { "_match" },
    { "destroy" },
    { "postblit" },

    { "LINE", "__LINE__" },
    { "FILE", "__FILE__" },
    { "DATE", "__DATE__" },
    { "TIME", "__TIME__" },
    { "TIMESTAMP", "__TIMESTAMP__" },
    { "VENDOR", "__VENDOR__" },
    { "VERSIONX", "__VERSION__" },
    { "EOFX", "__EOF__" },

    { "nan" },
    { "infinity" },
    { "dig" },
    { "epsilon" },
    { "mant_dig" },
    { "max_10_exp" },
    { "max_exp" },
    { "min_10_exp" },
    { "min_exp" },
    { "min_normal" },
    { "re" },
    { "im" },

    { "C" },
    { "D" },
    { "Windows" },
    { "Pascal" },
    { "System" },

    { "exit" },
    { "success" },
    { "failure" },

    { "keys" },
    { "values" },
    { "rehash" },

    { "sort" },
    { "reverse" },
    { "dup" },
    { "idup" },

    { "property" },
    { "safe" },
    { "trusted" },
    { "system" },
    { "disable" },

    // For inline assembler
    { "___out", "out" },
    { "___in", "in" },
    { "__int", "int" },
    { "__dollar", "$" },
    { "__LOCAL_SIZE" },

    // For operator overloads
    { "uadd",    "opPos" },
    { "neg",     "opNeg" },
    { "com",     "opCom" },
    { "add",     "opAdd" },
    { "add_r",   "opAdd_r" },
    { "sub",     "opSub" },
    { "sub_r",   "opSub_r" },
    { "mul",     "opMul" },
    { "mul_r",   "opMul_r" },
    { "div",     "opDiv" },
    { "div_r",   "opDiv_r" },
    { "mod",     "opMod" },
    { "mod_r",   "opMod_r" },
    { "eq",      "opEquals" },
    { "cmp",     "opCmp" },
    { "iand",    "opAnd" },
    { "iand_r",  "opAnd_r" },
    { "ior",     "opOr" },
    { "ior_r",   "opOr_r" },
    { "ixor",    "opXor" },
    { "ixor_r",  "opXor_r" },
    { "shl",     "opShl" },
    { "shl_r",   "opShl_r" },
    { "shr",     "opShr" },
    { "shr_r",   "opShr_r" },
    { "ushr",    "opUShr" },
    { "ushr_r",  "opUShr_r" },
    { "cat",     "opCat" },
    { "cat_r",   "opCat_r" },
    { "assign",  "opAssign" },
    { "addass",  "opAddAssign" },
    { "subass",  "opSubAssign" },
    { "mulass",  "opMulAssign" },
    { "divass",  "opDivAssign" },
    { "modass",  "opModAssign" },
    { "andass",  "opAndAssign" },
    { "orass",   "opOrAssign" },
    { "xorass",  "opXorAssign" },
    { "shlass",  "opShlAssign" },
    { "shrass",  "opShrAssign" },
    { "ushrass", "opUShrAssign" },
    { "catass",  "opCatAssign" },
    { "postinc", "opPostInc" },
    { "postdec", "opPostDec" },
    { "index",   "opIndex" },
    { "indexass", "opIndexAssign" },
    { "slice",   "opSlice" },
    { "sliceass", "opSliceAssign" },
    { "call",    "opCall" },
    { "cast",    "opCast" },
    { "match",   "opMatch" },
    { "next",    "opNext" },
    { "opIn" },
    { "opIn_r" },
    { "opStar" },
    { "opDot" },
    { "opDispatch" },
    { "opDollar" },
    { "opUnary" },
    { "opIndexUnary" },
    { "opSliceUnary" },
    { "opBinary" },
    { "opBinaryRight" },
    { "opOpAssign" },
    { "opIndexOpAssign" },
    { "opSliceOpAssign" },
    { "pow", "opPow" },
    { "pow_r", "opPow_r" },
    { "powass", "opPowAssign" },

    { "classNew", "new" },
    { "classDelete", "delete" },

    // For foreach
    { "apply", "opApply" },
    { "applyReverse", "opApplyReverse" },

    // Ranges
    { "Fempty", "empty" },
    { "Ffront", "front" },
    { "Fback", "back" },
    { "FpopFront", "popFront" },
    { "FpopBack", "popBack" },

    { "adDup", "_adDupT" },
    { "adReverse", "_adReverse" },

    // For internal functions
    { "aaLen", "_aaLen" },
    { "aaKeys", "_aaKeys" },
    { "aaValues", "_aaValues" },
    { "aaRehash", "_aaRehash" },
    { "monitorenter", "_d_monitorenter" },
    { "monitorexit", "_d_monitorexit" },
    { "criticalenter", "_d_criticalenter" },
    { "criticalexit", "_d_criticalexit" },
    { "_ArrayEq" },

    // For pragma's
    { "GNU_asm" },
    { "lib" },
    { "msg" },
    { "startaddress" },

#if IN_LLVM
    // LDC pragma's
    { "intrinsic" },
    { "va_intrinsic" },
    { "no_typeinfo" },
    { "no_moduleinfo" },
    { "Alloca", "alloca" },
    { "Shufflevector", "shufflevector" },
    { "vastart", "va_start" },
    { "vacopy", "va_copy" },
    { "vaend", "va_end" },
    { "vaarg", "va_arg" },
    { "ldc" },
    { "allow_inline" },
    { "llvm_inline_asm" },
    { "fence" },
    { "atomic_load" },
    { "atomic_store" },
    { "atomic_cmp_xchg" },
    { "atomic_rmw" },
#endif

    // For special functions
    { "tohash", "toHash" },
    { "tostring", "toString" },
    { "getmembers", "getMembers" },

    // Special functions
#if IN_DMD
    { "__alloca", "alloca" }, // has to be mapped because alloca is #defined if _MSC_VER
#endif
    { "main" },
    { "WinMain" },
    { "DllMain" },
    { "tls_get_addr", "___tls_get_addr" },

    // varargs implementation
    { "va_argsave_t", "__va_argsave_t" },
    { "va_argsave", "__va_argsave" },

    // Builtin functions
    { "std" },
    { "core" },
    { "math" },
    { "sin" },
    { "cos" },
    { "tan" },
    { "_sqrt", "sqrt" },
    { "_pow", "pow" },
    { "atan2" },
    { "rndtol" },
    { "expm1" },
    { "exp2" },
    { "yl2x" },
    { "yl2xp1" },
    { "fabs" },
    { "bitop" },
    { "bsf" },
    { "bsr" },
    { "bswap" },

    // Traits
    { "isAbstractClass" },
    { "isArithmetic" },
    { "isAssociativeArray" },
    { "isFinalClass" },
    { "isFloating" },
    { "isIntegral" },
    { "isScalar" },
    { "isStaticArray" },
    { "isUnsigned" },
    { "isVirtualFunction" },
    { "isVirtualMethod" },
    { "isAbstractFunction" },
    { "isFinalFunction" },
    { "isStaticFunction" },
    { "isRef" },
    { "isOut" },
    { "isLazy" },
    { "hasMember" },
    { "identifier" },
    { "parent" },
    { "getMember" },
    { "getOverloads" },
    { "getVirtualFunctions" },
    { "getVirtualMethods" },
    { "classInstanceSize" },
    { "allMembers" },
    { "derivedMembers" },
    { "isSame" },
    { "compiles" },
};


int main()
{
    FILE *fp;
    unsigned i;

    {
        fp = fopen("id.h","w");
        if (!fp)
        {   printf("can't open id.h\n");
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "// File generated by idgen.c\n");
#if __DMC__
        fprintf(fp, "#pragma once\n");
#endif
        fprintf(fp, "#ifndef DMD_ID_H\n");
        fprintf(fp, "#define DMD_ID_H 1\n");
        fprintf(fp, "struct Identifier;\n");
        fprintf(fp, "struct Id\n");
        fprintf(fp, "{\n");

        for (i = 0; i < sizeof(msgtable) / sizeof(msgtable[0]); i++)
        {   const char *id = msgtable[i].ident;

            fprintf(fp,"    static Identifier *%s;\n", id);
        }

        fprintf(fp, "    static void initialize();\n");
        fprintf(fp, "};\n");
        fprintf(fp, "#endif\n");

        fclose(fp);
    }

    {
        fp = fopen("id.c","w");
        if (!fp)
        {   printf("can't open id.c\n");
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "// File generated by idgen.c\n");
        fprintf(fp, "#include \"id.h\"\n");
        fprintf(fp, "#include \"identifier.h\"\n");
        fprintf(fp, "#include \"lexer.h\"\n");

        for (i = 0; i < sizeof(msgtable) / sizeof(msgtable[0]); i++)
        {   const char *id = msgtable[i].ident;
            const char *p = msgtable[i].name;

            if (!p)
                p = id;
            fprintf(fp,"Identifier *Id::%s;\n", id);
        }

        fprintf(fp, "void Id::initialize()\n");
        fprintf(fp, "{\n");

        for (i = 0; i < sizeof(msgtable) / sizeof(msgtable[0]); i++)
        {   const char *id = msgtable[i].ident;
            const char *p = msgtable[i].name;

            if (!p)
                p = id;
            fprintf(fp,"    %s = Lexer::idPool(\"%s\");\n", id, p);
        }

        fprintf(fp, "}\n");

        fclose(fp);
    }

    return EXIT_SUCCESS;
}
