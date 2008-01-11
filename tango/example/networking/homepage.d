
private import  tango.io.Console;

private import  tango.net.http.HttpClient,
                tango.net.http.HttpHeaders;

/*******************************************************************************

        Shows how to use HttpClient to retrieve content from the D website
        
*******************************************************************************/

void main()
{
        auto client = new HttpClient (HttpClient.Get, "http://www.digitalmars.com/d/intro.html");

        // open the client and get the input stream
        auto input = client.open;
        scope (exit)
               client.close;
        
        // display returned content on console
        if (client.isResponseOK)
            Cout.stream.copy (input);
        else
           Cout ("failed to return the D home page");  

        // flush the console
        Cout.newline;
}
