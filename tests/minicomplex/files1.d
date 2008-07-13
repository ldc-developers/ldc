module tangotests.files1;

//import tango.io.Stdout;
import tango.io.File;

void main()
{
    auto file = new File("files1.output");
    char[] str = "hello world from files1 test\n";
    void[] data = cast(void[])str;
    file.write(str);
}
