#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

#include "mars.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/GlobalValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Assembly/Writer.h"

#include "gen/logger.h"
#include "gen/irstate.h"

void Stream::writeType(std::ostream& OS, const llvm::Type& Ty) {
    llvm::raw_os_ostream raw(OS);
    llvm::WriteTypeSymbolic(raw, &Ty, gIR->module);
}

void Stream::writeValue(std::ostream& OS, const llvm::Value& V) {
    // Constants don't always get their types pretty-printed.
    // (Only treat non-global constants like this, so that e.g. global variables
    // still get their initializers printed)
    if (llvm::isa<llvm::Constant>(V) && !llvm::isa<llvm::GlobalValue>(V))
        llvm::WriteAsOperand(OS, &V, true, gIR->module);
    else
        OS << V;
}

namespace Logger
{
    static std::string indent_str;

    llvm::cl::opt<bool> _enabled("vv",
        llvm::cl::desc("Very verbose"),
        llvm::cl::ZeroOrMore);

    void indent()
    {
        if (_enabled) {
            indent_str += "* ";
        }
    }
    void undent()
    {
        if (_enabled) {
            assert(!indent_str.empty());
            indent_str.resize(indent_str.size()-2);
        }
    }
    Stream cout()
    {
        if (_enabled)
            return std::cout << indent_str;
        else
            return 0;
    }
    void println(const char* fmt,...)
    {
        if (_enabled) {
            printf("%s", indent_str.c_str());
            va_list va;
            va_start(va,fmt);
            vprintf(fmt,va);
            va_end(va);
            printf("\n");
        }
    }
    void print(const char* fmt,...)
    {
        if (_enabled) {
            printf("%s", indent_str.c_str());
            va_list va;
            va_start(va,fmt);
            vprintf(fmt,va);
            va_end(va);
        }
    }
    void enable()
    {
        _enabled = true;
    }
    void disable()
    {
        _enabled = false;
    }
    bool enabled()
    {
        return _enabled;
    }
    void attention(Loc loc, const char* fmt,...)
    {
        va_list va;
        va_start(va,fmt);
        vwarning(loc,fmt,va);
        va_end(va);
    }
}
