// adapted from dstress.run.a.associative_array_19_A to catch regressions early

module mini.aa7;

extern (C) int printf(char*, ...);

extern (C) void gc_collect();

union Key{
    char x;
}

class Payload {
    this(Key value) {
        value.x += 1;
        _value = value;
    }

    Key value() {
        return _value;
    }

    Key _value;
}

int main(){
    Payload[Key] aa;

    Key[] allKeys;
    static Key a = { 'a' };
    static Key b = { 'b' };
    static Key c = { 'c' };
    allKeys ~= a;
    allKeys ~= b;
    allKeys ~= c;

    foreach(Key key; allKeys) {
        aa[key] = new Payload(key);
    }

    int i = 0;
    foreach(Key key; allKeys) {
        printf("1st #%d\n", i++);
        assert(key in aa);
    }

    gc_collect();

    i = 0;
    foreach(Key key; allKeys) {
        printf("2nd #%d\n", i++);
        assert(key in aa);
    }

    return 0;
}

