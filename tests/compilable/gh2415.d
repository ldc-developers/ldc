// RUN: %ldc -c %s

struct Row
{
    float val;
    ref float opIndex(int i) { return val; }
}

struct Matrix
{
    Row row;
    ref Row opIndex(int i) { return row; }
}

void main()
{
    Matrix matrix;
    matrix[1][2] += 0.0;
}
