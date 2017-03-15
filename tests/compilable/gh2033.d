// RUN: %ldc -c %s

void foo(const(uint)[4] arg)
{
    // The front-end coerces the explicit and following implicit cast
    // (implicit: int[4] -> const(int[4])) into a single CastExp with
    // differing types CastExp::to and CastExp::type.
    // Make sure the repainting code performing the implicit cast
    // handles static arrays.
    const copy = cast(int[4]) arg;
}
