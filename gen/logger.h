#ifndef _llvmd_gen_logger_h_
#define _llvmd_gen_logger_h_

#include <iostream>

namespace Logger
{
    #ifndef LLVMD_NO_LOGGER
    void indent();
    void undent();
    std::ostream& cout();
    void println(const char* fmt,...);
    void print(const char* fmt,...);
    #else
    inline void indent() {}
    inline void undent() {}
    inline std::ostream& cout() { return std::cout; }
    inline void println(const char* fmt, ...) {}
    inline void print(const char* fmt, ...) {}
    #endif

    struct LoggerScope
    {
        LoggerScope()
        {
            #ifndef LLVMD_NO_LOGGER
            //std::cout << "-->indented\n";
            Logger::indent();
            #endif
            
        }
        ~LoggerScope()
        {
            #ifndef LLVMD_NO_LOGGER
            //std::cout << "<--undented\n";
            Logger::undent();
            #endif
        }
    };
}

#ifndef LLVMD_NO_LOGGER
#define LOG_SCOPE    Logger::LoggerScope _logscope;
#else
#define LOG_SCOPE
#endif

#endif
