module inputs.host_template_module;

template HostTemplate(T) {
    // Global variables are not allowed in @compute device code!
    // Without the fix, DCompute semantic analysis will process this
    // instantiation and erroneously emit an error for the global variable.
    int unsupportedGlobalVar = 42;
    
    void doHostThings() { }
}
