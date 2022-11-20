module imports.ldc_github_247a;
class ReadValue(T = int){
    void get_value(){ if( ctrl.active ){} }
}
class Value(T) : ReadValue!(T) {
    auto opAssign(T value){ // <<<=== change to Value!T to solve the ice
        return this;
    }
}
static Controller ctrl;
class Controller {
    bool active;
    Value!(int) pulse;
}
