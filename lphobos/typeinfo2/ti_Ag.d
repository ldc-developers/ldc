module typeinfo2.ti_Ag;

private int string_cmp(char[] s1, char[] s2)
{
    auto len = s1.length;
    if (s2.length < len)
        len = s2.length;
    int result = memcmp(s1.ptr, s2.ptr, len);
    if (result == 0)
        result = cast(int)(cast(ptrdiff_t)s1.length - cast(ptrdiff_t)s2.length);
    return result;
}

extern(C) int memcmp(void*,void*,size_t);

// byte[]

class TypeInfo_Ag : TypeInfo
{
    char[] toString() { return "byte[]"; }

    hash_t getHash(void *p)
    {	byte[] s = *cast(byte[]*)p;
	size_t len = s.length;
	byte *str = s.ptr;
	hash_t hash = 0;

	while (1)
	{
	    switch (len)
	    {
		case 0:
		    return hash;

		case 1:
		    hash *= 9;
		    hash += *cast(ubyte *)str;
		    return hash;

		case 2:
		    hash *= 9;
		    hash += *cast(ushort *)str;
		    return hash;

		case 3:
		    hash *= 9;
		    hash += (*cast(ushort *)str << 8) +
			    (cast(ubyte *)str)[2];
		    return hash;

		default:
		    hash *= 9;
		    hash += *cast(uint *)str;
		    str += 4;
		    len -= 4;
		    break;
	    }
	}

	return hash;
    }

    int equals(void *p1, void *p2)
    {
	byte[] s1 = *cast(byte[]*)p1;
	byte[] s2 = *cast(byte[]*)p2;

	return s1.length == s2.length &&
	       memcmp(cast(byte *)s1, cast(byte *)s2, s1.length) == 0;
    }

    int compare(void *p1, void *p2)
    {
	byte[] s1 = *cast(byte[]*)p1;
	byte[] s2 = *cast(byte[]*)p2;
	size_t len = s1.length;

	if (s2.length < len)
	    len = s2.length;
	for (size_t u = 0; u < len; u++)
	{
	    int result = s1[u] - s2[u];
	    if (result)
		return result;
	}
	return cast(int)s1.length - cast(int)s2.length;
    }

    size_t tsize()
    {
	return (byte[]).sizeof;
    }

    uint flags()
    {
	return 1;
    }

    TypeInfo next()
    {
	return typeid(byte);
    }
}


// ubyte[]

class TypeInfo_Ah : TypeInfo_Ag
{
    char[] toString() { return "ubyte[]"; }

    int compare(void *p1, void *p2)
    {
	char[] s1 = *cast(char[]*)p1;
	char[] s2 = *cast(char[]*)p2;

	return string_cmp(s1, s2);
    }

    TypeInfo next()
    {
	return typeid(ubyte);
    }
}

// void[]

class TypeInfo_Av : TypeInfo_Ah
{
    char[] toString() { return "void[]"; }

    TypeInfo next()
    {
	return typeid(void);
    }
}

// bool[]

class TypeInfo_Ab : TypeInfo_Ah
{
    char[] toString() { return "bool[]"; }

    TypeInfo next()
    {
	return typeid(bool);
    }
}

// char[]

class TypeInfo_Aa : TypeInfo_Ag
{
    char[] toString() { return "char[]"; }

    hash_t getHash(void *p)
    {	char[] s = *cast(char[]*)p;
	hash_t hash = 0;

version (all)
{
	foreach (char c; s)
	    hash = hash * 11 + c;
}
else
{
	size_t len = s.length;
	char *str = s;

	while (1)
	{
	    switch (len)
	    {
		case 0:
		    return hash;

		case 1:
		    hash *= 9;
		    hash += *cast(ubyte *)str;
		    return hash;

		case 2:
		    hash *= 9;
		    hash += *cast(ushort *)str;
		    return hash;

		case 3:
		    hash *= 9;
		    hash += (*cast(ushort *)str << 8) +
			    (cast(ubyte *)str)[2];
		    return hash;

		default:
		    hash *= 9;
		    hash += *cast(uint *)str;
		    str += 4;
		    len -= 4;
		    break;
	    }
	}
}
	return hash;
    }

    TypeInfo next()
    {
	return typeid(char);
    }
}


