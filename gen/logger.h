#ifndef _llvmd_gen_logger_h_
#define _llvmd_gen_logger_h_

#include <iostream>

struct Loc;

namespace Logger
{
    void indent();
    void undent();
    std::ostream& cout();
    void println(const char* fmt, ...);
    void print(const char* fmt, ...);
    void enable();
    void disable();
    bool enabled();

    void attention(const Loc& loc, const char* fmt, ...);

    struct LoggerScope
    {
        LoggerScope()
        {
            Logger::indent();
        }
        ~LoggerScope()
        {
            Logger::undent();
        }
    };
}

#define LOG_SCOPE    Logger::LoggerScope _logscope;

#endif

