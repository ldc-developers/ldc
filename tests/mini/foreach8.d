module mini.foreach8;

extern(C) int printf(char*, ...);

int main(){
    dchar[] array="\u2260";
    printf("array[0] == %u\n", array[0]);
    int test=0;
    int count=0;
    assert(count==0);
    foreach(int index, char obj; array){
        printf("%d\n", obj);
        test+=obj;
        count++;
    }
    assert(count==3);
    assert(test==0x20b);
    return 0;
}
