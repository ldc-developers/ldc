// $HeadURL: svn://svn.berlios.de/dstress/trunk/run/n/nested_class_03_A.d $
// $Date: 2005-06-18 09:15:32 +0200 (Sat, 18 Jun 2005) $
// $Author: thomask $

// @author@ John C <johnch_atms@hotmail.com>
// @date@   2005-06-09
// @uri@    news:d88vta$vak$1@digitaldaemon.com

//module dstress.run.n.nested_class_03_A;
module mini.nested17;

interface Inner{
    int value();        
}

class Outer{
    int x;

    Inner test(){
        printf("val = %d\n", x);
        return new class Inner {
            int y;

            this(){
                printf("val = %d\n", x);
                y=x;
            }

            int value(){
                return y;
            }
        };
    }
}

int main(){
    Outer o = new Outer();
    o.x=2;  
    int val = o.test().value();
    printf("val = %d\n", val);
    assert(val == o.x);
    return 0;
}

extern(C) int printf(char*, ...);
