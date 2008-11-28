#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "mars.h"

#include "gen/logger.h"

namespace Logger
{
    static std::string indent_str;
    static std::ofstream null_out("/dev/null");

    static bool _enabled = false;
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
    std::ostream& cout()
    {
        if (_enabled)
            return std::cout << indent_str;
        else
            return null_out;
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
    void attention(const Loc& loc, const char* fmt,...)
    {
        printf("Warning: %s: ", loc.toChars());
        va_list va;
        va_start(va,fmt);
        vprintf(fmt,va);
        va_end(va);
        printf("\n");
    }
}
