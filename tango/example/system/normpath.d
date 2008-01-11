/*****************************************************************

  Simple example that shows possible inputs to normalize and the
  corresponding outputs.

  Put into public domain by Lars Ivar Igesund.

*****************************************************************/

import tango.io.Stdout;

import tango.util.PathUtil;

int main()
{
version (Posix) {
    Stdout(normalize ( "/foo/../john")).newline;
    Stdout(normalize ( "foo/../john")).newline;    
    Stdout(normalize ( "foo/bar/..")).newline;    
    Stdout(normalize ( "foo/bar/../john")).newline;
    Stdout(normalize ( "foo/bar/doe/../../john")).newline;
    Stdout(normalize ( "foo/bar/doe/../../john/../bar")).newline;
    Stdout(normalize ( "./foo/bar/doe")).newline;
    Stdout(normalize ( "./foo/bar/doe/../../john/../bar")).newline;
    Stdout(normalize ( "./foo/bar/../../john/../bar")).newline;
    Stdout(normalize ( "foo/bar/./doe/../../john")).newline;
    Stdout(normalize ( "../../foo/bar/./doe/../../john")).newline;
    Stdout(normalize ( "../../../foo/bar")).newline;
    Stdout("** Should now throw exception as the following path is invalid for normalization.").newline;
    Stdout(normalize ( "/../../../foo/bar")).newline;
}
version (Windows) {
    Stdout(normalize ( "C:\\foo\\..\\john")).newline;
    Stdout(normalize ( "foo\\..\\john")).newline;    
    Stdout(normalize ( "foo\\bar\\..")).newline;    
    Stdout(normalize ( "foo\\bar\\..\\john")).newline;
    Stdout(normalize ( "foo\\bar\\doe\\..\\..\\john")).newline;
    Stdout(normalize ( "foo\\bar\\doe\\..\\..\\john\\..\\bar")).newline;
    Stdout(normalize ( ".\\foo\\bar\\doe")).newline;
    Stdout(normalize ( ".\\foo\\bar\\doe\\..\\..\\john\\..\\bar")).newline;
    Stdout(normalize ( ".\\foo\\bar\\..\\..\\john\\..\\bar")).newline;
    Stdout(normalize ( "foo\\bar\\.\\doe\\..\\..\\john")).newline;
    Stdout(normalize ( "..\\..\\foo\\bar\\.\\doe\\..\\..\\john")).newline;
    Stdout(normalize ( "..\\..\\..\\foo\\bar")).newline;
    Stdout("** Should now throw exception as the following path is invalid for normalization.").newline;
    Stdout(normalize ( "C:\\..\\..\\..\\foo\\bar")).newline;
}


    return 0;
}
