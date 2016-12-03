#include <cstddef>

struct Context;

extern "C" {
void* runtimecompile_modules_head = nullptr;
#ifdef _WIN32
__declspec(dllimport)
#endif
extern void rtCompileProcessImplSo(const void *modlist_head,
                                   const Context *context,
                                   std::size_t contextSize);

void rtCompileProcessImpl(const Context *context, std::size_t contextSize)
{
  rtCompileProcessImplSo(runtimecompile_modules_head, context, contextSize);
}
}
