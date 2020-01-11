// https://github.com/ldc-developers/ldc/issues/3272

// RUN: %ldc -g -run %s

enum E : void function()
{
    First = function() {}
}

void main()
{
    auto e = E.First;
    e();
}
