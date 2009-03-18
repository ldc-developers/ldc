#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "mars.h"

#include "llvm/Support/CommandLine.h"
#include "gen/logger.h"

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
    llvm::OStream cout()
    {
        if (_enabled)
            return llvm::cout << indent_str;
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
