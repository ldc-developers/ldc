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
import gen.semanticdcompute;

extern(C++) void dcomputeSemanticAnalysis(Module m);
extern(C++) int 

void extraLDCSpecificSemanticAnalysis(Modules modules)
{
    for (size_t i = 0; i < modules.dim; i++)
    {
        Module m = modules[i];
        if (hasComputeAttr(m)
            dcomputeSemanticAnalysis(m);
    }
    
}
