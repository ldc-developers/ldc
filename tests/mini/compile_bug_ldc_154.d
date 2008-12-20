template toInt(char[4] arg)
{
	const uint toInt = (cast(uint[]) arg)[0];
	}
	 
	 void main()
	 {
	   auto i = toInt!("abcd");
	   }
