#ifndef _llvmd_gen_logger_h_
#define _llvmd_gen_logger_h_

#include "llvm/Support/Streams.h"

struct Loc;

namespace Logger
{
    void indent();
    void undent();
    llvm::OStream cout();
    void println(const char* fmt, ...);
    void print(const char* fmt, ...);
    void enable();
    void disable();
    bool enabled();

    void attention(Loc loc, const char* fmt, ...);

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

#define IF_LOG       if (Logger::enabled())

#endif

