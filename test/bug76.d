module bug76;
char[] fmt(...)
{
    return "";
}
void main()
{
    char[] s = fmt();
}
