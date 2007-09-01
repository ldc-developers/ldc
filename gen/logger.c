#ifndef LLVMD_NO_LOGGER

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "logger.h"

namespace Logger
{
    static std::string indent_str;
    void indent()
    {
        indent_str += "  ";
    }
    void undent()
    {
        assert(!indent_str.empty());
        indent_str.resize(indent_str.size()-2);
    }
    std::ostream& cout()
    {
        return std::cout << indent_str;
    }
    void println(const char* fmt,...)
    {
        printf(indent_str.c_str());
        va_list va;
        va_start(va,fmt);
        vprintf(fmt,va);
        va_end(va);
        printf("\n");
    }
    void print(const char* fmt,...)
    {
        printf(indent_str.c_str());
        va_list va;
        va_start(va,fmt);
        vprintf(fmt,va);
        va_end(va);
    }
}

#endif
