#ifndef CALLBACK_OSTREAM_H
#define CALLBACK_OSTREAM_H

#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/STLExtras.h>

class CallbackOstream : public llvm::raw_ostream {
    using CallbackT = llvm::function_ref<void(const char*,size_t)>;
    CallbackT callback;
    uint64_t currentPos = 0;

    /// See raw_ostream::write_impl.
    void write_impl(const char *Ptr, size_t Size) override;

    uint64_t current_pos() const override;
public:
    explicit CallbackOstream(CallbackT c);
};

#endif // CALLBACK_OSTREAM_H
