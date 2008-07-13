module bug15;

bool bool1(bool b) {
    if (b) return true;
    else return false;
}

bool bool2(bool b) {
    if (b) {return true;}
    else {return false;}
}

void main()
{
    assert(bool1(true));
    assert(!bool2(false));
}
