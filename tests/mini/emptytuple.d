template T(B...) {
	typedef B TArgs;
}

int main(){
	alias T!() instantiate;
	return 0;
}
