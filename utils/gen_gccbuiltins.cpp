//===-- gen_gccbuiltins.cpp - GCC builtin module generator ----------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This tool reads the GCC builtin definitions from LLVM's Intrinsics.td for
// a given architecture and accordingly generates a ldc.gccbuiltins_<arch>
// module for using them from D code.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Main.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <assert.h>
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace std;
using namespace llvm;

string dtype(Record* rec, bool readOnlyMem)
{
    Init* typeInit = rec->getValueInit("VT");
    if(!typeInit)
        return "";

    string type = typeInit->getAsString();

    if(type == "iPTR")
        return readOnlyMem ? "const void*" : "void*";

    string vec = "";

    if(type[0] == 'v')
    {
        size_t i = 1;
        while(i != type.size() && type[i] <= '9' && type[i] >= '0')
            i++;

        vec = type.substr(1, i - 1);
        type = type.substr(i);
    }

    if(vec.size() > 0 && type.size() > 0)
    {
        int typeSize, vecElements;
        if(
            sscanf(vec.c_str(), "%d", &vecElements) == 1 &&
            sscanf(type.c_str() + 1, "%d", &typeSize) == 1 &&
            typeSize * vecElements > 256)
        {
            return "";
        }
    }

    if(type == "i8")
        return "byte" + vec;
    else if(type == "i16")
        return "short" + vec;
    else if(type == "i32")
        return "int" + vec;
    else if(type == "i64")
        return "long" + vec;
    else if(type == "f32")
        return "float" + vec;
    else if(type == "f64")
        return "double" + vec;
    else
        return "";
}

string attributes(ListInit* propertyList)
{
    string prop =
#if LDC_LLVM_VER >= 307
        propertyList->size()
#else
         propertyList->getSize()
#endif
        ? propertyList->getElementAsRecord(0)->getName() : "";

    return
        prop == "IntrNoMem" ? " pure @safe" :
        prop == "IntrReadArgMem" ? " pure" :
        prop == "IntrReadWriteArgMem" ? " pure" : "";
}

void processRecord(raw_ostream& os, Record& rec, string arch)
{
    if(!rec.getValue("GCCBuiltinName"))
        return;

    string builtinName = rec.getValueAsString("GCCBuiltinName");
    string name =  rec.getName();

    if(name.substr(0, 4) != "int_" || name.find(arch) == string::npos)
        return;

    name = name.substr(4);
    replace(name.begin(), name.end(), '_', '.');
    name = string("llvm.") + name;

    ListInit* propsList = rec.getValueAsListInit("Properties");
    string prop =
#if LDC_LLVM_VER >= 307
        propsList->size()
#else
        propsList->getSize()
#endif
        ? propsList->getElementAsRecord(0)->getName() : "";

    bool readOnlyMem = prop == "IntrReadArgMem" || prop == "IntrReadMem";

    ListInit* paramsList = rec.getValueAsListInit("ParamTypes");
    vector<string> params;
    for(unsigned int i = 0; i <
#if LDC_LLVM_VER >= 307
        paramsList->size();
#else
        paramsList->getSize();
#endif
        i++)
    {
        string t = dtype(paramsList->getElementAsRecord(i), readOnlyMem);
        if(t == "")
            return;

        params.push_back(t);
    }

    ListInit* retList = rec.getValueAsListInit("RetTypes");
    string ret;
#if LDC_LLVM_VER >= 307
    size_t sz = retList->size();
#else
    size_t sz = retList->getSize();
#endif
    if(sz == 0)
        ret = "void";
    else if(sz == 1)
    {
        ret = dtype(retList->getElementAsRecord(0), false);
        if(ret == "")
            return;
    }
    else
        return;

    os << "pragma(LDC_intrinsic, \"" + name + "\")\n    ";
    os << ret + " " + builtinName + "(";

    if(params.size())
        os << params[0];

    for(size_t i = 1; i < params.size(); i++)
        os << ", " << params[i];

    os << ")" + attributes(rec.getValueAsListInit("Properties")) + ";\n\n";
}

std::string arch;

bool emit(raw_ostream& os, RecordKeeper& records)
{
    os << "module ldc.gccbuiltins_";
    os << arch;
    os << "; \n\nimport core.simd;\n\nnothrow @nogc:\n\n";

#if LDC_LLVM_VER >= 306
    const auto &defs = records.getDefs();
#else
    map<string, Record*> defs = records.getDefs();
#endif

    for (const auto& d : defs)
        processRecord(os, *d.second, arch);

    return false;
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        fprintf(stderr, "There must be exactly two command line arguments\n");
        return 1;
    }

    llvm::SmallString<128> file(LLVM_INTRINSIC_TD_PATH);
    sys::path::append(file, "llvm");
    sys::path::append(file, "IR");
    sys::path::append(file, "Intrinsics.td");

    string iStr = string("-I=") + string(LLVM_INTRINSIC_TD_PATH);
    string oStr = string("-o=") + argv[1];

    vector<char*> args2(argv, argv + 1);
    args2.push_back(const_cast<char*>(file.c_str()));
    args2.push_back(const_cast<char*>(iStr.c_str()));
    args2.push_back(const_cast<char*>(oStr.c_str()));

    cl::ParseCommandLineOptions(args2.size(), &args2[0]);
    arch = argv[2];
    return TableGenMain(argv[0], emit);
}
