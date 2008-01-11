; ModuleID = 'wrap.bc'
@errno = external global i32		; <i32*> [#uses=2]

define i32 @getErrno() {
entry:
	%tmp = load i32* @errno		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @setErrno(i32 %val) {
entry:
	store i32 %val, i32* @errno
	ret i32 %val
}
