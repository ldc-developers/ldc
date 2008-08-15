// $HeadURL: svn://svn.berlios.de/dstress/trunk/run/b/bug_glue_700_A.d $
// $Date: 2007-02-27 17:42:34 +0100 (Di, 27 Feb 2007) $
// $Author: thomask $

// @author@	Kevin Bealer <kevinbealer@gmail.com>
// @date@	2007-01-22
// @uri@	http://d.puremagic.com/issues/show_bug.cgi?id=875
// @desc@	[Issue 875] crash in glue.c line 700

module dstress.run.b.bug_glue_700_A;

template T(B...) {
	typedef B TArgs;
}

int main(){
	alias T!() instantiate;
	return 0;
}
