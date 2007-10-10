#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "gen/logger.h"

namespace Logger
{
    static std::string indent_str;
    static std::ofstream null_out("/dev/null");

    static bool enabled = false;
    void indent()
    {
        if (enabled)
        indent_str += "  ";
    }
    void undent()
    {
        if (enabled) {
            assert(!indent_str.empty());
            indent_str.resize(indent_str.size()-2);
        }
    }
    std::ostream& cout()
    {
        if (enabled)
            return std::cout << indent_str;
        else
            return null_out;
    }
    void println(const char* fmt,...)
    {
        if (enabled) {
            printf(indent_str.c_str());
            va_list va;
            va_start(va,fmt);
            vprintf(fmt,va);
            va_end(va);
            printf("\n");
        }
    }
    void print(const char* fmt,...)
    {
        if (enabled) {
            printf(indent_str.c_str());
            va_list va;
            va_start(va,fmt);
            vprintf(fmt,va);
            va_end(va);
        }
    }
    void enable()
    {
        enabled = true;
    }
    void disable()
    {
        enabled = false;
    }
}

