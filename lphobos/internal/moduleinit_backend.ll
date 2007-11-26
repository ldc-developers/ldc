; ModuleID = 'internal.moduleinit_backend'

@_d_moduleinfo_array = appending constant [1 x i8*] [ i8* null ]

define i8** @_d_get_moduleinfo_array() {
entry:
        %tmp = getelementptr [1 x i8*]* @_d_moduleinfo_array, i32 0, i32 0
        ret i8** %tmp
}
