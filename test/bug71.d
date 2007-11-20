module bug71;

void main()
{
    static TypeInfo skipCI(TypeInfo valti)
    {
      while (1)
      {
    if (valti.classinfo.name.length == 18 &&
        valti.classinfo.name[9..18] == "Invariant")
        valti = (cast(TypeInfo_Invariant)valti).next;
    else if (valti.classinfo.name.length == 14 &&
        valti.classinfo.name[9..14] == "Const")
        valti = (cast(TypeInfo_Const)valti).next;
    else
        break;
      }
      return valti;
    }
}