// slightly modified chunk of dmd-testsuite's compilable/interpret3.d

// RUN: %ldc -c %s

struct Vector13831()
{
}

struct Coord13831
{
    union // reversed field order
    {
        Vector13831!() vector;
        struct { short x; }
    }
}

struct Chunk13831
{
    this(Coord13831)
    {
        coord = coord;
    }

    Coord13831 coord;

    static const Chunk13831* unknownChunk = new Chunk13831(Coord13831());
}
