import std.variant;

class C
{
    Algebraic!int[string][] W;

    auto foo()
    {
        auto d = {
            foreach (w; W)
                w["path"].get!string;
        };
    }
}
