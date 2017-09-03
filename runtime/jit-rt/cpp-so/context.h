#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstddef> //size_t

// must be synchronized with D source
typedef void (*InterruptPointHandlerT)(void*, const char* action, const char* object);
typedef void (*FatalHandlerT)(void*, const char* reason);
typedef void (*DumpHandlerT)(void*, const char* str, std::size_t len);

#pragma pack(push,1)

struct Context final
{
    unsigned optLevel = 0;
    unsigned sizeLevel = 0;
    InterruptPointHandlerT interruptPointHandler = nullptr;
    void* interruptPointHandlerData = nullptr;
    FatalHandlerT fatalHandler = nullptr;
    void* fatalHandlerData = nullptr;
    DumpHandlerT dumpHandler = nullptr;
    void* dumpHandlerData = nullptr;
};
#pragma pack(pop)

#endif // CONTEXT_H
