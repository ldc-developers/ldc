// RUN: %ldc -run %s

extern(C) __gshared string[] rt_options = [ "key=value" ];

void main()
{
    import rt.config;
    assert(rt_configOption("key") == "value");
}
