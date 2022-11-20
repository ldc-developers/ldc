struct OperandFormatDescriptor
{
    string name;
}

enum OperandFormat
{
    SrcDst = OperandFormatDescriptor("SrcDst"),
    DstSrc = OperandFormatDescriptor("DstSrc")
}

struct OpcodeDescriptor
{
    OperandFormat operandFormat;
}

enum Opcodes
{
    Load = OpcodeDescriptor(OperandFormat.DstSrc),
}

void main()
{
    assert(OperandFormat.init.name == "SrcDst");
    assert(OpcodeDescriptor.init.operandFormat.name == "SrcDst");
    assert(Opcodes.Load.operandFormat.name == "DstSrc");
}
