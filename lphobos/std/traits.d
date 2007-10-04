module std.traits;
struct TypeHolder(S, T...) {
  S _ReturnType;
  T _ParameterTypeTuple;
}
TypeHolder!(S, T) *IFTI_gen(S, T...)(S delegate(T) dg) { return null; }
TypeHolder!(S, T) *IFTI_gen(S, T...)(S function(T) dg) { return null; }
template ParameterTypeTuple(T) {
  alias typeof(IFTI_gen(T.init)._ParameterTypeTuple) ParameterTypeTuple;
}
template ReturnType(T) {
  alias typeof(IFTI_gen(T.init)._ReturnType) ReturnType;
}
template isArray(T) { const bool isArray=false; }
template isArray(T: T[]) { const bool isArray=true; }
