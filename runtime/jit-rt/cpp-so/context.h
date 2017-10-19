#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstddef> //size_t

// must be synchronized with D source
enum class DumpStage : int { OriginalIR = 0, OptimizedIR = 1, FinalAsm = 2 };

typedef void (*InterruptPointHandlerT)(void *, const char *action,
                                       const char *object);
typedef void (*FatalHandlerT)(void *, const char *reason);
typedef void (*DumpHandlerT)(void *, DumpStage stage, const char *str,
                             std::size_t len);

struct Context final {
  unsigned optLevel = 0;
  unsigned sizeLevel = 0;
  InterruptPointHandlerT interruptPointHandler = nullptr;
  void *interruptPointHandlerData = nullptr;
  FatalHandlerT fatalHandler = nullptr;
  void *fatalHandlerData = nullptr;
  DumpHandlerT dumpHandler = nullptr;
  void *dumpHandlerData = nullptr;
};

#endif // CONTEXT_H
