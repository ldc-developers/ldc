/*******************************************************************************

        Shows how to create a basic socket client, and how to converse with
        a remote server. The server must be running for this to succeed

*******************************************************************************/

private import  tango.io.Console;

private import  tango.net.SocketConduit, 
                tango.net.InternetAddress;

void main()
{
        // make a connection request to the server
        auto request = new SocketConduit;
        request.connect (new InternetAddress ("localhost", 8080));
        request.output.write ("hello\n");

        // wait for response (there is an optional timeout supported)
        char[64] response;
        auto size = request.input.read (response);

        // close socket
        request.close;

        // display server response
        Cout (response[0..size]).newline;
}
