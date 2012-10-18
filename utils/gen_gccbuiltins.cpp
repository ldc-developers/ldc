#include <map>
#include <string>
#include <algorithm>

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <llvm/TableGen/Main.h>
#if LDC_LLVM_VER < 302
#include <llvm/TableGen/TableGenAction.h>
#endif
#include <llvm/TableGen/Record.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PathV1.h>
#include <llvm/ADT/StringRef.h>

using namespace std;
using namespace llvm;

string dtype(Record* rec)
{
    Init* typeInit = rec->getValueInit("VT");
    if(!typeInit)
        return "";

    string type = typeInit->getAsString();

    if(type == "iPTR")
        return "void*";

    string vec = "";

    if(type[0] == 'v')
    {
        size_t i = 1;
        while(i != type.size() && type[i] <= '9' && type[i] >= '0')
            i++;

        vec = type.substr(1, i - 1);
        type = type.substr(i);
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

    ListInit* paramsList = rec.getValueAsListInit("ParamTypes");
    vector<string> params;
    for(int i = 0; i < paramsList->getSize(); i++)
    {
        string t = dtype(paramsList->getElementAsRecord(i));
        if(t == "")
            return;

        params.push_back(t);
    }

    ListInit* retList = rec.getValueAsListInit("RetTypes");
    string ret;
    if(retList->getSize() == 0)
        ret = "void";
    else if(retList->getSize() == 1)
    {
        ret = dtype(retList->getElementAsRecord(0));
        if(ret == "")
            return;
    }
    else
        return;

    os << "pragma(intrinsic, \"" + name + "\")\n    ";
    os << ret + " " + builtinName + "(";

    if(params.size())
        os << params[0];

    for(int i = 1; i < params.size(); i++)
        os << ", " << params[i];

    os << ");\n\n";
}

std::string arch;

bool emit(raw_ostream& os, RecordKeeper& records)
{
    os << "module ldc.gccbuiltins_";
    os << arch;
    os << "; \n\nimport core.simd;\n\n";

    map<string, Record*> defs = records.getDefs();

    for(
        map<string, Record* >::iterator it = defs.begin();
        it != defs.end();
        it++)
    {
        processRecord(os, *(*it).second, arch);
    }

    return false;
}

#if LDC_LLVM_VER < 302
struct ActionImpl : TableGenAction
{
    bool operator()(raw_ostream& os, RecordKeeper& records)
    {
        return emit(os, records);
    }
};
#endif

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        fprintf(stderr, "There must be exactly two command line arguments\n");
        return 1;
    }

    sys::Path file(LLVM_INTRINSIC_TD_PATH);
    file.appendComponent("llvm");
    file.appendComponent("Intrinsics.td");

    string iStr = string("-I=") + string(LLVM_INTRINSIC_TD_PATH);
    string oStr = string("-o=") + argv[1];

    vector<char*> args2(argv, argv + 1);
    args2.push_back(const_cast<char*>(file.c_str()));
    args2.push_back(const_cast<char*>(iStr.c_str()));
    args2.push_back(const_cast<char*>(oStr.c_str()));

    cl::ParseCommandLineOptions(args2.size(), &args2[0]);
    arch = argv[2];
#if LDC_LLVM_VER >= 302
    return TableGenMain(argv[0], emit);
#else
    ActionImpl act;
    return TableGenMain(argv[0], act);
#endif
}
