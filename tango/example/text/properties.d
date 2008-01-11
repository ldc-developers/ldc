private import  tango.io.Buffer,
                tango.io.Console;

private import  tango.text.Properties;

/*******************************************************************************

        Illustrates simple usage of tango.text.Properties

*******************************************************************************/

void main() 
{
        char[][char[]] aa;
        aa ["foo"] = "something";
        aa ["bar"] = "something else";
        aa ["wumpus"] = "";

        // write associative-array to a buffer; could use a file
        auto props = new Properties!(char);
        auto buffer = new Buffer (256);
        props.save (buffer, aa);

        // reset and repopulate AA from the buffer
        aa = null;
        props.load (buffer, (char[] name, char[] value){aa[name] = value;});

        // display result
        foreach (name, value; aa)
                 Cout (name) (" = ") (value).newline;
}

