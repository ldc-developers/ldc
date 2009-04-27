// simple example that shows off getting D wrappers from C values.
module llvmsample3;

import llvm.c.Core;
import llvm.llvm;

void main()
{
    auto m = new Module("sample3");

    // global int32
    auto gi = m.addGlobal(Type.Int32, "myint");
    gi.initializer = ConstantInt.GetU(Type.Int32, 42);

    // this is not a cached value, it's recreated dynamically
    auto _i = gi.initializer;
    auto ci = cast(ConstantInt)_i;
    assert(ci !is null);
    ci.dump;

    // global struct
    auto st = StructType.Get([Type.Double,Type.Double,Type.Double]);
    auto gs = m.addGlobal(st, "mystruct");
    auto elems = new Constant[3];
    foreach(i,ref e; elems)
        e = ConstantReal.Get(Type.Double, i+1);
    gs.initializer = ConstantStruct.Get(elems);

    // again this is not a cached value.
    auto s = gs.initializer;
    auto cs = cast(ConstantStruct)s;
    assert(cs !is null);

    cs.dump;
}
