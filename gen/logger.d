//===-- gen/logger.d - Codegen debug logging ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// D implementation of the functionality of logger.{h/cpp}.
//
//===----------------------------------------------------------------------===//

module gen.logger;

private extern (C++) extern __gshared bool _Logger_enabled;
extern (C++, Logger)
{
    void indent();
    void undent();
    void printIndentation();
}

struct Log
{
    static bool enabled()
    {
        return _Logger_enabled;
    }

    static void indent()
    {
        if (enabled())
            .indent();
    }

    static void undent()
    {
        if (enabled())
            .undent();
    }

    // Usage:  auto _ = Log.newScope();
    static auto newScope()
    {
        struct ScopeExitUndenter
        {
            ~this()
            {
                Logger.undent();
            }
        }

        Logger.indent();
        return ScopeExitUndenter();
    }

    static void printfln(T...)(T args)
    {
        static import std.stdio;

        if (enabled())
        {
            printIndentation();
            std.stdio.writefln(args);
        }
    }

    static void printf(T...)(T args)
    {
        static import std.stdio;

        if (enabled())
        {
            printIndentation();
            std.stdio.writef(args);
        }
    }
}
