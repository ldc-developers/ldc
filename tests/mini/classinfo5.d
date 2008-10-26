extern(C) int printf(char*, ...);

class BaseClass {} 
 
void main() 
{ 
    Object o_cr = BaseClass.classinfo.create();
    Object o_new = new BaseClass;
    printf("CIaddr: %X\n", cast(size_t*)BaseClass.classinfo);
    printf("Create: %X\n", cast(size_t*)o_cr.classinfo);
    printf("New:    %X\n", cast(size_t*)o_new.classinfo);
    assert(cast(size_t*)o_cr.classinfo == cast(size_t*)o_new.classinfo);
} 
