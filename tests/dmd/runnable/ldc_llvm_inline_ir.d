pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);

alias inlineIR!(`
    %rp = alloca i32
    %ip = alloca i32
    store i32 1, i32* %rp
    store i32 %0, i32* %ip
    %cond = icmp sgt i32 %0, 0
    br i1 %cond, label %loop, label %end

    loop:
        %i = load i32, i32* %ip
        %r = load i32, i32* %rp
        %rnext = mul i32 %r, %i
        %inext = sub i32 %i, 1
        store i32 %rnext, i32* %rp
        store i32 %inext, i32* %ip
        %cond1 = icmp sgt i32 %inext, 0
        br i1 %cond1, label %loop, label %end

    end:
        %ret = load i32, i32* %rp
        ret i32 %ret`,
    int, int) factorial;

alias __vector(int[4]) int4;

alias inlineIR!(`
    %ret = shufflevector <4 x i32> %0, <4 x i32> %1, <4 x i32> <i32 4, i32 1, i32 7, i32 6>
    ret <4 x i32> %ret`,
    int4, int4, int4) shuffle;

alias inlineIR!(`store i16 %0, i16* %1`, void, short, short*) store;

alias inlineIR!(`
    %cmp = fcmp olt double %0, %1
    %ret = sext i1 %cmp to i64
    ret i64 %ret`,
    long, double, double) lt; 

alias __vector(float[4]) float4;
alias inlineIR!(`
    %r = fadd <4 x float> %0, %1
    ret <4 x float> %r`, float4, float4, float4) foo;

alias inlineIR!(`store i32 %1, i32* %0`, void, int*, int) bar;

void main()
{
    assert(factorial(6) == 720);
    
    int4 va = [0, 10, 20, 30];
    int4 vb = [40, 50, 60, 70];
    assert(shuffle(va, vb).array == [40, 10, 70, 60]);
    
    short a = 42;
    short b = 0;
    store(a, &b);
    assert(b == 42);
    
    assert(lt(0, 1) == -1);
    assert(lt(1, 0) == 0);
}

