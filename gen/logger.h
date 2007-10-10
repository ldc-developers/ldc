#ifndef _llvmd_gen_logger_h_
#define _llvmd_gen_logger_h_

#include <iostream>

namespace Logger
{
    void indent();
    void undent();
    std::ostream& cout();
    void println(const char* fmt, ...);
    void print(const char* fmt, ...);
    void enable();
    void disable();
    

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

