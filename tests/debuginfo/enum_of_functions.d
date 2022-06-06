// https://github.com/ldc-developers/ldc/issues/3272

// RUN: %ldc -g -run %s

enum F : void function()
{
    First = function() {}
}

enum D : void delegate()
{
    First = () {}
}

void main()
{
    auto f = F.First;
    f();

    auto d = D.First;
    d();
}
