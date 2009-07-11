//===- StripMetaData - Strips D-specific metadata -------------------------===//
//
//                             The LLVM D Compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// There's an issue with the new LLVM metadata support; an assertion fires when
// trying to generate asm for metadata globals.
//
// This pass is a workaround; it deletes the metadata LDC generates so the code
// generator doesn't see it.
// Obviously, this should only run after all passes that make use of that
// metadata or they won't work.
//
//===----------------------------------------------------------------------===//

#include "gen/metadata.h"

#define DEBUG_TYPE "strip-metadata"

#include "Passes.h"

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumDeleted, "Number of metadata globals deleted");

//===----------------------------------------------------------------------===//
// StripMetaData Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
    /// This pass optimizes library functions from the D runtime as used by LDC.
    ///
    class VISIBILITY_HIDDEN StripMetaData : public ModulePass {
        public:
        static char ID; // Pass identification
        StripMetaData() : ModulePass(&ID) {}
        
        bool runOnModule(Module &M);
    };
    char StripMetaData::ID = 0;
} // end anonymous namespace.

static RegisterPass<StripMetaData>
X("strip-metadata", "Delete D-specific metadata");

// Public interface to the pass.
ModulePass *createStripMetaData() {
  return new StripMetaData(); 
}

/// runOnFunction - Top level algorithm.
///
bool StripMetaData::runOnModule(Module &M) {
    bool Changed = false;
    for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E;) {
        GlobalVariable* G = I++;
        if (G->getNameLen() >= 9 && !strncmp(G->getNameStart(), "llvm.ldc.", 9)) {
            assert(G->hasInitializer() && isa<MDNode>(G->getInitializer())
                && "Not a metadata global?");
            Changed = true;
            NumDeleted++;
            DEBUG(DOUT << "Deleting " << *G << '\n');
            G->eraseFromParent();
        }
    }
    return Changed;
}
