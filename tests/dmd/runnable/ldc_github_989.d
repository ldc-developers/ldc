struct sockaddr
{
    ushort   sa_family;
    byte[14] sa_data;
}

struct sockaddr_in
{
    ushort sin_family;
    ushort sin_port;
    uint   sin_addr;
    ubyte[16 - ushort.sizeof -
          ushort.sizeof - uint.sizeof] __pad;
}

struct in6_addr
{
    union
    {
        ubyte[16] s6_addr;
        ushort[8] s6_addr16;
        uint[4] s6_addr32;
    }
}

struct sockaddr_in6
{
    ushort sin6_family;
    ushort   sin6_port;
    uint    sin6_flowinfo;
    in6_addr    sin6_addr;
    uint    sin6_scope_id;
}

struct NetworkAddress {
    // if this union is removed, the segfault disappears.
    union {
        sockaddr addr;
        sockaddr_in addr_ip4;
        sockaddr_in6 addr_ip6;
    }
}

interface InputStream { }

interface OutputStream { }

// if ': InputStream, OutputStream' is removed,
//  the segfault moves from cast_.d line 71 to cast_.d line 59
interface Stream : InputStream, OutputStream { }

interface TCPConnection : Stream { }

class Libevent2TCPConnection : TCPConnection {
    // if m_localAddress or m_removeAddress is removed,
    //  the segfault disappears
    NetworkAddress m_localAddress, m_remoteAddress;
}

void main() {
    // use auto or Libevent2TCPConnection instead of TCPConnection
    //  and the segfault disappears.
    TCPConnection conn = new Libevent2TCPConnection();
    Stream s = conn;
} 