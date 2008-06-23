module tangotests.arrays4;

struct Str { int a,b; }
void main() {
    Str[] arr = new Str[64];

    auto tmp = Str(1,2);
    arr[] = tmp;
    assert(arr[0].a == 1);
    assert(arr[0].b == 2);
    assert(arr[13].a == 1);
    assert(arr[13].b == 2);
    assert(arr[42].a == 1);
    assert(arr[42].b == 2);
    assert(arr[63].a == 1);
    assert(arr[63].b == 2);

    arr[] = Str(3,4);
    assert(arr[0].a == 3);
    assert(arr[0].b == 4);
    assert(arr[13].a == 3);
    assert(arr[13].b == 4);
    assert(arr[42].a == 3);
    assert(arr[42].b == 4);
    assert(arr[63].a == 3);
    assert(arr[63].b == 4);
}
