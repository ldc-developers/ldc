module bug66;
import std.stdio;
class Scene { string name() { return "Scene"; } }
class Group : Scene { this () { } }
void main() { writefln((new Group).name); }
