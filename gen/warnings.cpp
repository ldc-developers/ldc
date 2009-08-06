#include "mars.h"
#include "mtype.h"
#include "expression.h"

#include "gen/warnings.h"

void warnInvalidPrintfCall(Loc loc, Expression* arguments, size_t nargs)
{
    Expression* arg = arguments;

    // make sure first argument is a string literal, or we can't do much
    // TODO make it smarter ?
    if (arg->op != TOKstring)
        return; // assume valid

    StringExp* strexp = (StringExp*)arg;

    // not wchar or dhar
    if (strexp->sz != 1)
    {
        warning(loc, "printf does not support wchar and dchar strings");
        return;
    }

#if 0
    // check the format string
    const char* str = (char*)strexp->string;
    for (size_t i = 0; i < strexp->len; ++i)
    {
        // TODO
    }
#endif
}
