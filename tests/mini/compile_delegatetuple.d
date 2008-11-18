alias char[] string;
template Unstatic(T) { alias T Unstatic; }
template Unstatic(T: T[]) { alias T[] Unstatic; }
template StupleMembers(T...) {
  static if (T.length) {
      const int id=T[0..$-1].length;
      const string str=StupleMembers!(T[0..$-1]).str~"Unstatic!(T["~id.stringof~"]) _"~id.stringof~"; ";
  } else const string str="";
}

struct Stuple(T...) {
 mixin(StupleMembers!(T).str);
}     
Stuple!(string, void delegate(float)) foo;
