// RUN: %ldc -c %s

struct Vector
{
    float x, y, z;
}

struct QAngle
{
    float x, y, z;

    QAngle opOpAssign(string op)(const(QAngle))
    {
        return this;
    }
}

void OnUserCmdPre()
{
    Vector ss;
    QAngle dd;
    dd -= cast(QAngle)ss;
}
