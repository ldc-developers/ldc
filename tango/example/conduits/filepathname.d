import tango.io.Console;

import tango.io.FilePath;

void main(){
    Cout ((new FilePath(r"d:\path\foo.bat")).name).newline;
    Cout ((new FilePath(r"d:\path.two\bar")).name).newline;
    Cout ((new FilePath("/home/user.name/bar.")).name).newline;
    Cout ((new FilePath(r"d:\path.two\bar")).name).newline;
    Cout ((new FilePath("/home/user/.resource")).name).newline;
}
