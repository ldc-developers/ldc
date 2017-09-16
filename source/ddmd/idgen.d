/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _idgen.d)
 */

// Program to generate string files in d data structures.
// Saves much tedious typing, and eliminates typo problems.
// Generates:
//      id.h
//      id.c

import core.stdc.stdio;
import core.stdc.stdlib;

struct Msgtable
{
    const(char)* ident;      // name to use in DMD source
    const(char)* name;       // name in D executable
};

Msgtable[] msgtable =
[
    { "IUnknown" },
    { "Object" },
    { "object" },
    { "string" },
    { "wstring" },
    { "dstring" },
    { "max" },
    { "min" },
    { "This", "this" },
    { "_super", "super" },
    { "ctor", "__ctor" },
    { "dtor", "__dtor" },
    { "__xdtor", "__xdtor" },
    { "__fieldDtor", "__fieldDtor" },
    { "__aggrDtor", "__aggrDtor" },
    { "postblit", "__postblit" },
    { "__xpostblit", "__xpostblit" },
    { "__fieldPostblit", "__fieldPostblit" },
    { "__aggrPostblit", "__aggrPostblit" },
    { "classInvariant", "__invariant" },
    { "unitTest", "__unitTest" },
    { "require", "__require" },
    { "ensure", "__ensure" },
    { "_init", "init" },
    { "__sizeof", "sizeof" },
    { "__xalignof", "alignof" },
    { "_mangleof", "mangleof" },
    { "stringof" },
    { "_tupleof", "tupleof" },
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
    { "RTInfo" },
    { "Throwable" },
    { "Error" },
    { "withSym", "__withSym" },
    { "result", "__result" },
    { "returnLabel", "__returnLabel" },
    { "line" },
    { "empty", "" },
    { "p" },
    { "q" },
    { "__vptr" },
    { "__monitor" },
    { "gate", "__gate" },
    { "__c_long" },
    { "__c_ulong" },
    { "__c_long_double" },
    { "cpp_type_info_ptr", "__cpp_type_info_ptr" },
    { "_assert", "assert" },
    { "_unittest", "unittest" },

    { "TypeInfo" },
    { "TypeInfo_Class" },
    { "TypeInfo_Interface" },
    { "TypeInfo_Struct" },
    { "TypeInfo_Enum" },
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
    { "destroy" },
    { "xopEquals", "__xopEquals" },
    { "xopCmp", "__xopCmp" },
    { "xtoHash", "__xtoHash" },

    { "LINE", "__LINE__" },
    { "FILE", "__FILE__" },
    { "MODULE", "__MODULE__" },
    { "FUNCTION", "__FUNCTION__" },
    { "PRETTY_FUNCTION", "__PRETTY_FUNCTION__" },
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
    { "Objective" },

    { "exit" },
    { "success" },
    { "failure" },

    { "keys" },
    { "values" },
    { "rehash" },

    { "sort" },
    { "reverse" },

    { "property" },
    { "nogc" },
    { "safe" },
    { "trusted" },
    { "system" },
    { "disable" },

    // For inline assembler
    { "___out", "out" },
    { "___in", "in" },
    { "__int", "int" },
    { "_dollar", "$" },
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
    { "_cast",    "opCast" },
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
    { "_ArrayPostblit" },
    { "_ArrayDtor" },

    // For pragma's
    { "Pinline", "inline" },
    { "lib" },
    { "mangle" },
    { "msg" },
    { "startaddress" },

    // For special functions
    { "tohash", "toHash" },
    { "tostring", "toString" },
    { "getmembers", "getMembers" },

    // Special functions
    { "__alloca", "alloca" },
    { "main" },
    { "WinMain" },
    { "DllMain" },
    { "tls_get_addr", "___tls_get_addr" },
    { "entrypoint", "__entrypoint" },
    { "rt_init" },
    { "__cmp" },

    // varargs implementation
    { "va_start" },

    // Builtin functions
    { "std" },
    { "core" },
    { "attribute" },
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
    { "isTemplate" },
    { "isPOD" },
    { "isNested" },
    { "isFloating" },
    { "isIntegral" },
    { "isScalar" },
    { "isStaticArray" },
    { "isUnsigned" },
    { "isVirtualFunction" },
    { "isVirtualMethod" },
    { "isAbstractFunction" },
    { "isFinalFunction" },
    { "isOverrideFunction" },
    { "isStaticFunction" },
    { "isRef" },
    { "isOut" },
    { "isLazy" },
    { "hasMember" },
    { "identifier" },
    { "getProtection" },
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
    { "parameters" },
    { "getAliasThis" },
    { "getAttributes" },
    { "getFunctionAttributes" },
    { "getUnitTests" },
    { "getVirtualIndex" },
    { "getPointerBitmap" },

    // For C++ mangling
    { "allocator" },
    { "basic_string" },
    { "basic_istream" },
    { "basic_ostream" },
    { "basic_iostream" },
    { "char_traits" },

    // Compiler recognized UDA's
    { "udaSelector", "selector" },

    // IN_LLVM: LDC-specific pragmas.
    { "LDC_intrinsic" },
    { "LDC_no_typeinfo" },
    { "LDC_no_moduleinfo" },
    { "LDC_alloca" },
    { "LDC_va_start" },
    { "LDC_va_copy" },
    { "LDC_va_end" },
    { "LDC_va_arg" },
    { "LDC_verbose" },
    { "LDC_allow_inline" },
    { "LDC_never_inline" },
    { "LDC_inline_asm" },
    { "LDC_inline_ir" },
    { "LDC_fence" },
    { "LDC_atomic_load" },
    { "LDC_atomic_store" },
    { "LDC_atomic_cmp_xchg" },
    { "LDC_atomic_rmw" },
    { "LDC_global_crt_ctor" },
    { "LDC_global_crt_dtor" },
    { "LDC_extern_weak" },
    { "LDC_profile_instr" },

    // IN_LLVM: LDC-specific traits.
    { "targetCPU" },
    { "targetHasFeature" },
    
    // IN_LLVM: LDC-specific attributes
    { "ldc" },
    { "attributes" },
    { "udaAllocSize", "allocSize" },
    // fastmath is an AliasSeq of llvmAttr and llvmFastMathFlag
    { "udaOptStrategy", "optStrategy" },
    { "udaLLVMAttr", "llvmAttr" },
    { "udaLLVMFastMathFlag", "llvmFastMathFlag" },
    { "udaSection", "section" },
    { "udaTarget", "target" },
    { "udaWeak", "_weak" },
    { "udaCompute", "compute" },
    { "udaKernel", "_kernel" },
    
    // IN_LLVM: DCompute specific types and functionss
    { "dcompute" },
    { "dcPointer", "Pointer" },
    { "dcReflect", "__dcompute_reflect" },
];


int main()
{
    {
        auto fp = fopen("ddmd/id.h","wb");
        if (!fp)
        {
            printf("can't open ddmd/id.h\n");
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "// File generated by idgen.d\n");
        fprintf(fp, "#ifndef DMD_ID_H\n");
        fprintf(fp, "#define DMD_ID_H 1\n");
        fprintf(fp, "class Identifier;\n");
        fprintf(fp, "struct Id\n");
        fprintf(fp, "{\n");

        foreach(e; msgtable)
        {
            auto id = e.ident;
            fprintf(fp,"    static Identifier *%s;\n", id);
        }

        fprintf(fp, "    static void initialize();\n");
        fprintf(fp, "};\n");
        fprintf(fp, "#endif\n");

        fclose(fp);
    }

    {
        auto fp = fopen("ddmd/id.d","wb");
        if (!fp)
        {
            printf("can't open ddmd/id.d\n");
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "// File generated by idgen.d\n");
        fprintf(fp, "\n");
        fprintf(fp, "module ddmd.id;\n");
        fprintf(fp, "\n");
        fprintf(fp, "import ddmd.identifier, ddmd.tokens;\n");
        fprintf(fp, "\n");
        fprintf(fp, "struct Id\n");
        fprintf(fp, "{\n");

        foreach(e; msgtable)
        {
            auto id = e.ident;
            auto p = e.name;

            if (!p)
                p = id;
            fprintf(fp, "    extern (C++) static __gshared Identifier %s;\n", id);
        }

        fprintf(fp, "\n");
        fprintf(fp, "    extern (C++) static void initialize()\n");
        fprintf(fp, "    {\n");

        foreach(e; msgtable)
        {
            auto id = e.ident;
            auto p = e.name;

            if (!p)
                p = id;
            fprintf(fp,"        %s = Identifier.idPool(\"%s\");\n", id, p);
        }

        fprintf(fp, "    }\n");
        fprintf(fp, "}\n");

        fclose(fp);
    }

    return EXIT_SUCCESS;
}
