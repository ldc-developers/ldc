private import  tango.io.Console;

private import  tango.net.http.HttpGet;

/*******************************************************************************

        Read a page from a website, gathering the entire page before 
        returning any content. This illustrates a high-level approach
        to retrieving web-content, whereas the homepage example shows
        a somewhat lower-level approach. 

        Note that this expects a fully qualified URL (with scheme), 
        such as "http://www.digitalmars.com/d/intro.html"

*******************************************************************************/

void main (char[][] args)
{
	char[] url = (args.length is 2) ? args[1] : "http://www.digitalmars.com/d/intro.html";
            
        // open a web-page for reading (see HttpPost for writing)
        auto page = new HttpGet (url);

        // retrieve and flush display content
        Cout (cast(char[]) page.read) ();
}

