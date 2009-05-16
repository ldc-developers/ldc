#ifndef _llvmd_gen_logger_h_
#define _llvmd_gen_logger_h_

#include "llvm/Support/Streams.h"

#ifndef IS_PRINTF
# ifdef __GNUC__
#  define IS_PRINTF(FMTARG) __attribute((__format__ (__printf__, (FMTARG), (FMTARG)+1) ))
# else
#  define IS_PRINTF(FMTARG)
# endif
#endif

struct Loc;

namespace Logger
{
    void indent();
    void undent();
    llvm::OStream cout();
    void println(const char* fmt, ...) IS_PRINTF(1);
    void print(const char* fmt, ...) IS_PRINTF(1);
    void enable();
    void disable();
    bool enabled();

    void attention(Loc loc, const char* fmt, ...) IS_PRINTF(2);

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

