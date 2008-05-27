module tangotests.mem6;

extern(C) int printf(char*,...);

int main(){
        int[] index;
        char[] value;

        foreach(int i, char c; "_\U00012345-"){
                printf("str[%d] = %d\n", i , cast(int)c);
                index ~= i;
                //value ~= c;
        }
        printf("done\n");

        return 0;
}
