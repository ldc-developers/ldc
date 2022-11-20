interface IBase {}

class Impl(T): IBase {
    T value;
}
void main() {
    IBase x = new Impl!uint;
    IBase y = new Impl!string;
    IBase z = new Impl!double;

    IBase[string] fldFormats1 = [
        "Key": x,
        "Name": y,
        "Percent": z
    ];

    IBase[string] fldFormats2 = cast(IBase[string])[
        "Key": new Impl!uint,
        "Name": new Impl!string,
        "Percent": new Impl!double
    ];
}
