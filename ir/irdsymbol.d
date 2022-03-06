module ir.irdsymbol;

extern (C++) struct IrDsymbol
{
    void* irData;
    int m_type;  // enum
    int m_state; // enum

    this(const ref IrDsymbol);
    ~this();
    void doRegister();
}
