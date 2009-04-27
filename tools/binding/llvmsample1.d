// simple hello world sample of D LLVM
module llvmsample1;

import llvm.llvm;

void main()
{
    // create module
    auto m = new Module("sample1");
    scope(exit) m.dispose();

    // declare string
    auto chello = ConstantArray.GetString("Hello World!\n", true);
    auto hello = m.addGlobal(chello.type, "hellostring");
    hello.initializer = chello;
    hello.linkage = Linkage.Internal;
    hello.globalConstant = true;

    // declare printf
    auto printfType = FunctionType.Get(Type.Int32, [ PointerType.Get(Type.Int8) ], true);
    auto llprintf = m.addFunction(printfType, "printf");

    // declare main
    auto mainType = FunctionType.Get(Type.Int32, null);
    auto llmain = m.addFunction(mainType, "main");

    // create builder
    auto b = new Builder;
    scope(exit) b.dispose();

    // create main body block
    auto bb = llmain.appendBasicBlock("entry");
    b.positionAtEnd(bb);

    // call printf
    auto zero = ConstantInt.GetU(Type.Int32, 0);
    auto helloptr = b.buildGEP(hello, [ zero, zero ], "str");
    helloptr.dump();
    auto args = [ helloptr ];
    auto call = b.buildCall(llprintf, args, "");

    // return 0
    b.buildRet(ConstantInt.GetS(Type.Int32, 0));

    // write bitcode
    m.writeBitcodeToFile("sample1.bc");
}
