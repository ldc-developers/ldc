#ifndef VALUEPARSER_H
#define VALUEPARSER_H

namespace llvm {
class Constant;
class Type;
class DataLayout;
}

struct Context;

llvm::Constant *parseInitializer(const Context &context,
                                 const llvm::DataLayout &dataLayout,
                                 llvm::Type *type,
                                 const void *data);


#endif // VALUEPARSER_H
