
#include "callback_ostream.h"

void CallbackOstream::write_impl(const char *Ptr, size_t Size) {
  callback(Ptr, Size);
  currentPos += Size;
}

uint64_t CallbackOstream::current_pos() const { return currentPos; }

CallbackOstream::CallbackOstream(CallbackOstream::CallbackT c) : callback(c) {
  SetUnbuffered();
}
