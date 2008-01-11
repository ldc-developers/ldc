/**

    Example that shows how the format specifiers can be used to index into
    the argument list.

    Put into public domain by Lars Ivar Igesund.

*/

import tango.io.Stdout;

void main(){
    Stdout.formatln("Many {1} can now be {0} around to make {2} easier,\n and {1} can also be repeated.", 
                    "switched", "arguments", "localization");
}
