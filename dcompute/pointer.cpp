//===-- pointer.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/pointer.h"
#include "llvm/Pass.h"
#include <string>
namespace  {
class PointerReplacePass : ModulePass {
  //%d is an uint addrspace template arg
  //%t is mangled template Type parameter
  //%s is a string template parameter 2a or 2b (* and + respectivly)
  static std::string MangledPrefix = "_D8dcompute5types7pointer18__T7PointerVki";
  static std::string OpUnaryStar = "_D8dcompute5types7pointer18__T7PointerVki%dT"
                      "%tZ7Pointer21__T7opUnaryVAyaa1_2aZ7opUnaryMFNaNbNcNiNeZ%t";
  static std::string OpIndex = "_D8dcompute5types7pointer18__T7PointerVk%d2T"
                      "%tZ7Pointer7opIndexMFNaNbNcNiNemZ%t";
  static std::string OpBinary ="_D8dcompute5types7pointer18__T7PointerVki%dT"
                  "%tZ7Pointer22__T8opBinaryVAyaa1_%sZ8opBinaryMFNaNbNiNelZ"
                  "S8dcompute5types7pointer18__T7PointerVk%d2T%tZ7Pointer";
  unsigned mapping[PSnum];
  void handleOpUnaryStar(llvm::Function *f,int realAddrspace);
  void handleOpIndex(llvm::Function *f,int realAddrspace);
  void handleOpBinary(llvm::Function *f,int realAddrspace,bool plus);
  void handleOpOpassign(llvm::Function *f,int realAddrspace bool plus);
public:
  static char ID;
  PointerReplacePass(unsigned _mapping[PSnum]) : ModulePass(ID) {
    for (int i=0; i < PSnum; i++) {
        mapping[i] = _mapping[i];
    }
  }
  bool runOnModule(Module& m);
};
}

ModulePass * createPointerReplacePass(int mapping[PSnum]) {
  return new PointerReplacePass(mapping);
}
bool PointerReplacePass::runOnModule(Module& m) {
  for (llvm::Function* f : m.functions()) {
    auto name = f->getName();
    if (name.startsWith(MangledPrefix)) {
      int pseudoAddrspace = char(name[MangledPrefix.length()]) - '0';
      int realAddrspace = mapping[pseudoAddrspace];
      size_t pos;
      // TODO: skip the prefix
      if ((pos = name.find("Z7Pointer21__T7opUnaryVAyaa1_2aZ7opUnaryMFNaNbNcNiNeZ")) != string::npos) {
        handleOpUnaryStar(f,realAddrspace);
      } else if ((pos = name.find("Z7Pointer7opIndexMFNaNbNcNiNemZ"))!= string::npos) {
        handleOpIndex(f,realAddrspace);
      } else if ((pos = name.find("7Pointer22__T8opBinaryVAyaa1_")) != string::npos){
        //bool plus =
        handleOpBinary(f,realAddrspace,plus);
      }
        //TODO: OpOpAssign
    }
  }
}
