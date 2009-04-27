// simple test of recursive types.
module llvmsample2;

import llvm.llvm;

void main()
{
    auto th = new TypeHandle();
    auto s = StructType.Get([ PointerType.Get(th.resolve) ], false);
    th.refine(s);
    s.dump();
    th.dispose();

    auto t = getTypeOf(s.ll);
    t.dump();

    assert(s is t);
}
