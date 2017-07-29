/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _errors.d)
 */

module ddmd.errors;

import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.string;
import core.sys.posix.unistd;
import core.sys.windows.windows;
import ddmd.globals;
import ddmd.root.outbuffer;
import ddmd.root.rmem;

version (Windows) extern (C) int isatty(int);

enum COLOR : int
{
    COLOR_BLACK     = 0,
    COLOR_RED       = 1,
    COLOR_GREEN     = 2,
    COLOR_BLUE      = 4,
    COLOR_YELLOW    = COLOR_RED | COLOR_GREEN,
    COLOR_MAGENTA   = COLOR_RED | COLOR_BLUE,
    COLOR_CYAN      = COLOR_GREEN | COLOR_BLUE,
    COLOR_WHITE     = COLOR_RED | COLOR_GREEN | COLOR_BLUE,
}

alias COLOR_BLACK = COLOR.COLOR_BLACK;
alias COLOR_RED = COLOR.COLOR_RED;
alias COLOR_GREEN = COLOR.COLOR_GREEN;
alias COLOR_BLUE = COLOR.COLOR_BLUE;
alias COLOR_YELLOW = COLOR.COLOR_YELLOW;
alias COLOR_MAGENTA = COLOR.COLOR_MAGENTA;
alias COLOR_CYAN = COLOR.COLOR_CYAN;
alias COLOR_WHITE = COLOR.COLOR_WHITE;

version (Windows)
{
    extern (C++) static WORD consoleAttributes(HANDLE h)
    {
        static __gshared CONSOLE_SCREEN_BUFFER_INFO sbi;
        static __gshared bool sbi_inited = false;
        if (!sbi_inited)
            sbi_inited = GetConsoleScreenBufferInfo(h, &sbi) != FALSE;
        return sbi.wAttributes;
    }

    enum : int
    {
        FOREGROUND_WHITE = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
    }
}

extern (C++) bool isConsoleColorSupported()
{
    version (CRuntime_DigitalMars)
    {
        return isatty(stderr._file) != 0;
    }
    else version (CRuntime_Microsoft)
    {
        return isatty(fileno(stderr)) != 0;
    }
    else version (Posix)
    {
        const(char)* term = getenv("TERM");
        return isatty(STDERR_FILENO) && term && term[0] && 0 != strcmp(term, "dumb");
    }
    else
    {
        return false;
    }
}

extern (C++) void setConsoleColorBright(bool bright)
{
    version (Windows)
    {
        HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
        WORD attr = consoleAttributes(h);
        SetConsoleTextAttribute(h, attr | (bright ? FOREGROUND_INTENSITY : 0));
    }
    else
    {
        fprintf(stderr, "\033[%dm", bright ? 1 : 0);
    }
}

extern (C++) void setConsoleColor(COLOR color, bool bright)
{
    version (Windows)
    {
        HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
        WORD attr = consoleAttributes(h);
        attr = (attr & ~(FOREGROUND_WHITE | FOREGROUND_INTENSITY)) | ((color & COLOR_RED) ? FOREGROUND_RED : 0) | ((color & COLOR_GREEN) ? FOREGROUND_GREEN : 0) | ((color & COLOR_BLUE) ? FOREGROUND_BLUE : 0) | (bright ? FOREGROUND_INTENSITY : 0);
        SetConsoleTextAttribute(h, attr);
    }
    else
    {
        fprintf(stderr, "\033[%d;%dm", bright ? 1 : 0, 30 + cast(int)color);
    }
}

extern (C++) void resetConsoleColor()
{
    version (Windows)
    {
        HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
        SetConsoleTextAttribute(h, consoleAttributes(h));
    }
    else
    {
        fprintf(stderr, "\033[m");
    }
}

/**************************************
 * Print error message
 */
extern (C++) void error(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    verror(loc, format, ap);
    va_end(ap);
}

extern (C++) void error(Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    verror(loc, format, ap);
    va_end(ap);
}

extern (C++) void error(const(char)* filename, uint linnum, uint charnum, const(char)* format, ...)
{
    Loc loc;
    loc.filename = filename;
    loc.linnum = linnum;
    loc.charnum = charnum;
    va_list ap;
    va_start(ap, format);
    verror(loc, format, ap);
    va_end(ap);
}

extern (C++) void errorSupplemental(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    verrorSupplemental(loc, format, ap);
    va_end(ap);
}

extern (C++) void warning(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    vwarning(loc, format, ap);
    va_end(ap);
}

extern (C++) void warningSupplemental(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    vwarningSupplemental(loc, format, ap);
    va_end(ap);
}

extern (C++) void deprecation(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    vdeprecation(loc, format, ap);
    va_end(ap);
}

extern (C++) void deprecationSupplemental(const ref Loc loc, const(char)* format, ...)
{
    va_list ap;
    va_start(ap, format);
    vdeprecation(loc, format, ap);
    va_end(ap);
}

// Just print, doesn't care about gagging
extern (C++) void verrorPrint(const ref Loc loc, COLOR headerColor, const(char)* header, const(char)* format, va_list ap, const(char)* p1 = null, const(char)* p2 = null)
{
    const p = loc.toChars();
    if (global.params.color)
        setConsoleColorBright(true);
    if (*p)
        fprintf(stderr, "%s: ", p);
    mem.xfree(cast(void*)p);
    if (global.params.color)
        setConsoleColor(headerColor, true);
    fputs(header, stderr);
    if (global.params.color)
        resetConsoleColor();
    if (p1)
        fprintf(stderr, "%s ", p1);
    if (p2)
        fprintf(stderr, "%s ", p2);
    OutBuffer tmp;
    tmp.vprintf(format, ap);
    fprintf(stderr, "%s\n", tmp.peekString());
    fflush(stderr);
}

// header is "Error: " by default (see errors.h)
extern (C++) void verror(const ref Loc loc, const(char)* format, va_list ap, const(char)* p1 = null, const(char)* p2 = null, const(char)* header = "Error: ")
{
    global.errors++;
    if (!global.gag)
    {
        verrorPrint(loc, COLOR_RED, header, format, ap, p1, p2);
        if (global.errorLimit && global.errors >= global.errorLimit)
            fatal(); // moderate blizzard of cascading messages
    }
    else
    {
        if (global.params.showGaggedErrors)
        {
            fprintf(stderr, "(spec:%d) ", global.gag);
            verrorPrint(loc, COLOR_MAGENTA, header, format, ap, p1, p2);
        }
        global.gaggedErrors++;
    }
}

// Doesn't increase error count, doesn't print "Error:".
extern (C++) void verrorSupplemental(const ref Loc loc, const(char)* format, va_list ap)
{
    COLOR color;
    if (global.gag)
    {
        if (!global.params.showGaggedErrors)
            return;
        color = COLOR_MAGENTA;
    }
    else
        color = COLOR_RED;
    verrorPrint(loc, color, "       ", format, ap);
}

extern (C++) void vwarning(const ref Loc loc, const(char)* format, va_list ap)
{
    if (global.params.warnings && !global.gag)
    {
        verrorPrint(loc, COLOR_YELLOW, "Warning: ", format, ap);
        //halt();
        if (global.params.warnings == 1)
            global.warnings++; // warnings don't count if gagged
    }
}

extern (C++) void vwarningSupplemental(const ref Loc loc, const(char)* format, va_list ap)
{
    if (global.params.warnings && !global.gag)
        verrorPrint(loc, COLOR_YELLOW, "       ", format, ap);
}

extern (C++) void vdeprecation(const ref Loc loc, const(char)* format, va_list ap, const(char)* p1 = null, const(char)* p2 = null)
{
    static __gshared const(char)* header = "Deprecation: ";
    if (global.params.useDeprecated == 0)
        verror(loc, format, ap, p1, p2, header);
    else if (global.params.useDeprecated == 2 && !global.gag)
        verrorPrint(loc, COLOR_BLUE, header, format, ap, p1, p2);
}

extern (C++) void vdeprecationSupplemental(const ref Loc loc, const(char)* format, va_list ap)
{
    if (global.params.useDeprecated == 0)
        verrorSupplemental(loc, format, ap);
    else if (global.params.useDeprecated == 2 && !global.gag)
        verrorPrint(loc, COLOR_BLUE, "       ", format, ap);
}

/***************************************
 * Call this after printing out fatal error messages to clean up and exit
 * the compiler.
 */
extern (C++) void fatal()
{
    version (none)
    {
        halt();
    }
    exit(EXIT_FAILURE);
}

/**************************************
 * Try to stop forgetting to remove the breakpoints from
 * release builds.
 */
extern (C++) void halt()
{
    assert(0);
}
