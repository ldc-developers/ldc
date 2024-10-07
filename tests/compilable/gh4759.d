// RUN: %ldc -c %s

void func(void delegate())
{
    enum ident(alias F) = __traits(identifier, __traits(parent, F));
    func({ enum i = ident!({}); });
}
