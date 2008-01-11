/*******************************************************************************

        Shows how to create a basic socket server, and how to talk to
        it from a socket client. Note that both the server and client
        are entirely simplistic, and therefore this is for illustration
        purposes only. See HttpServer for something more robust.

*******************************************************************************/

private import  tango.core.Thread;

private import  tango.io.Console;

private import  tango.net.ServerSocket,
                tango.net.SocketConduit;

/*******************************************************************************

        Create a socket server, and have it respond to a request

*******************************************************************************/

void main()
{
        const int port = 8080;
 
        // thread body for socket-listener
        void run()
        {       
                auto server = new ServerSocket (new InternetAddress(port));
                
                // wait for requests
                auto request = server.accept;

                // write a response 
                request.output.write ("server replies 'hello'");
        }

        // start server in a separate thread, and wait for it to start
        (new Thread (&run)).start;
        Thread.sleep (0.250);

        // make a connection request to the server
        auto request = new SocketConduit;
        request.connect (new InternetAddress("localhost", port));

        // wait for and display response (there is an optional timeout)
        char[64] response;
        auto len = request.input.read (response);
        Cout (response[0..len]).newline;

        request.close;
}
