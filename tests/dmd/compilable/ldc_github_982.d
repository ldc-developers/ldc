import std.datetime.stopwatch;
void main() {
    auto r = cast(Duration[2])benchmark!({},{})(1);
}