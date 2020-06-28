// RUN: %ldc -betterC -run %s

extern (C) int main()
{
    int a = 5;
    scope(exit) a = 6;
    create(0, 1, "2");
    return 0;
}

void create(uint a, uint b, string c) {
}
