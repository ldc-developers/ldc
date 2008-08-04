module internal.contract;

import std.string: toString;
extern(C):

void exit(int);

/*void _d_assert(bool cond, uint line, char[] msg)
{
    if (!cond) {
        printf("Aborted(%u): %.*s\n", line, msg.length, msg.ptr);
        exit(1);
    }
}*/
void _d_assert(string file, uint line) {
  throw new Exception(file~":"~.toString(line)~": Assertion failed!");
}

void _d_assert_msg(string msg, string file, uint line) {
  throw new Exception(file~": "~.toString(line)~": Assertion failed: \""~msg~"\"");
}
