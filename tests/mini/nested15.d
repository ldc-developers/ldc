// $HeadURL: svn://svn.berlios.de/dstress/trunk/run/t/this_13_A.d $
// $Date: 2006-12-31 20:59:08 +0100 (Sun, 31 Dec 2006) $
// $Author: thomask $

// @author@	Frank Benoit <benoit@tionex.de>
// @date@	2006-10-09
// @uri@	http://d.puremagic.com/issues/show_bug.cgi?id=419
// @desc@	[Issue 419] New: Anonymous classes are not working.

// added to mini to catch regressions earlier

module mini.nested15;

class I {
	abstract void get( char[] s );
}

class C{
	void init(){
		I i = new class() I {
			void get( char[] s ){
				func();
			}
		};
	}
	void func( ){ }
}

int main(){
	C c = new C();
	c.init();

	return 0;
}

