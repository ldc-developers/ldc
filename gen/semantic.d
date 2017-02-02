//===-- gen/semantic.d - Additional LDC semantic analysis ---------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module gen.semantic;

import ddmd.arraytypes;
import ddmd.dsymbol;
import ddmd.dmodule;

extern(C++) void dcomputeSemanticAnalysis(Module m);
extern(C++) int hasComputeAttr(Dsymbol m);

extern(C++) void extraLDCSpecificSemanticAnalysis(ref Modules modules)
{
    for (size_t i = 0; i < modules.dim; i++)
    {
        Module m = modules[i];
        if (hasComputeAttr(m))
            dcomputeSemanticAnalysis(m);
    }
    
}