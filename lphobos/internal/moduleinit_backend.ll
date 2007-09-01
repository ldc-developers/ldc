; ModuleID = 'internal.moduleinit_backend'
@_d_module_ctor_array = appending global [1 x void ()*] zeroinitializer
@_d_module_dtor_array = appending global [1 x void ()*] zeroinitializer

define void ()** @_d_get_module_ctors() {
entry:
        %tmp = getelementptr [1 x void ()*]* @_d_module_ctor_array, i32 0, i32 0
        ret void ()** %tmp
}

define void ()** @_d_get_module_dtors() {
entry:
        %tmp = getelementptr [1 x void ()*]* @_d_module_dtor_array, i32 0, i32 0
        ret void ()** %tmp
}
