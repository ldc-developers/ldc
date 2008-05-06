char* toStringz (char[] s)
{
        if (s.ptr)
            if (! (s.length && s[$-1] is 0))
                   s = s ~ '\0';
        return s.ptr;
}

