//===-- gen/semantic.d - Additional LDC semantic analysis ---------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module gen.semantic;

import dmd.arraytypes;
import dmd.dsymbol;
import dmd.dmodule;

extern(C++) void dcomputeSemanticAnalysis(Module m);
extern(C) int hasComputeAttr(Dsymbol m);
extern(C++) void runAllSemanticAnalysisPlugins(Module m);

extern(C++) void extraLDCSpecificSemanticAnalysis(ref Modules modules)
{
    // First finish DCompute SemA for all modules, before calling plugins.
    foreach(m; modules[])
    {
        if (hasComputeAttr(m)) {
            dcomputeSemanticAnalysis(m);
        }
    }

    foreach(m; modules[])
    {
        runAllSemanticAnalysisPlugins(m);
    }
}
