// Based on dstress.run.t.tuple_15_A;

module tuple_and_vararg;

template TypeTuple(TList...){
	alias TList TypeTuple;
}

void main(){
	auto y = function(TypeTuple!(uint,uint) ab, ...){};
        y(1, 2);
        y(1, 2, "foo", 3.0);
}
