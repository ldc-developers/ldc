module bug7;

class love { }
void bug() {
    love[] child;
    child.length=1;
    assert(child[0] is null);
    child[0]=null;
    assert(child[0] is null);
}
void main()
{
    bug();
}
