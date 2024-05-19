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

#define BUILTIN_NAME_STRING "ClangBuiltinName"

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

    if(type == "i1" && vec.empty())
        return "bool";
    else if(type == "i8")
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

StringRef attributes(ListInit* propertyList)
{
  const auto prop = propertyList->size()
                    ? propertyList->getElementAsRecord(0)->getName()
                    : "";

  if (prop == "IntrNoMem")
    return " pure @safe";
  if (prop == "IntrReadArgMem" || prop == "IntrReadWriteArgMem")
    return " pure";
  return "";
}

void processRecord(raw_ostream& os, Record& rec, string arch)
{
    if(!rec.getValue(BUILTIN_NAME_STRING))
        return;

    const StringRef builtinName = rec.getValueAsString(BUILTIN_NAME_STRING);
    string name = rec.getName().str();

    if(name.substr(0, 4) != "int_" || name.find(arch) == string::npos)
        return;

    name = name.substr(4);
    replace(name.begin(), name.end(), '_', '.');
    name = string("llvm.") + name;

    ListInit* propsList = rec.getValueAsListInit("IntrProperties");
    const StringRef prop =
        propsList->size() ? propsList->getElementAsRecord(0)->getName() : "";

    bool readOnlyMem = prop == "IntrReadArgMem" || prop == "IntrReadMem";

    ListInit* paramsList = rec.getValueAsListInit("ParamTypes");
    vector<string> params;
    for(unsigned int i = 0; i < paramsList->size(); i++)
    {
        string t = dtype(paramsList->getElementAsRecord(i), readOnlyMem);
        if(t == "")
            return;

        params.push_back(t);
    }

    ListInit* retList = rec.getValueAsListInit("RetTypes");
    string ret;
    size_t sz = retList->size();
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

    os << "pragma(LDC_intrinsic, \"" << name << "\")\n    ";
    os << ret << " " << builtinName << "(";

    if(params.size())
        os << params[0];

    for(size_t i = 1; i < params.size(); i++)
        os << ", " << params[i];

    os << ")" << attributes(propsList) << ";\n\n";
}

std::string arch;

bool emit(raw_ostream& os, RecordKeeper& records)
{
    os << "module ldc.gccbuiltins_";
    os << arch;
    os << "; \n\nimport core.simd;\n\nnothrow @nogc:\n\n";

    const auto &defs = records.getDefs();

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

    string iStr = string("-I=") + LLVM_INTRINSIC_TD_PATH;
    string oStr = string("-o=") + argv[1];

    vector<char*> args2(argv, argv + 1);
    args2.push_back(const_cast<char*>(file.c_str()));
    args2.push_back(const_cast<char*>(iStr.c_str()));
    args2.push_back(const_cast<char*>(oStr.c_str()));

    cl::ParseCommandLineOptions(args2.size(), &args2[0]);
    arch = argv[2];
    return TableGenMain(argv[0], emit);
}
