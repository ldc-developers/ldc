// adapted from dstress.run.a.associative_array_19_<n> to catch regressions early

module mini.aa7;

extern (C) int printf(char*, ...);
extern (C) void gc_collect();


int main(){
    char*[char] aa;

    char key = 'a';
    aa[key] = &key;
    gc_collect();
    assert(aa[key] == &key);
    assert(key in aa);

    return 0;
}

