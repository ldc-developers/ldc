/*******************************************************************************

        Tokenize input from the console. There are a variety of handy
        tokenizers in the tango.text package ~ this illustrates usage
        of an iterator that recognizes quoted-strings within an input
        array, and splits elements on a provided set of delimiters

*******************************************************************************/

import tango.io.Console;

import Text = tango.text.Util;
  
void main()
{
        // flush the console output, since we have no newline present
        Cout ("Please enter some space-separated tokens: ") ();

        // create quote-aware iterator for handling space-delimited
        // tokens from the console input
        foreach (element; Text.quotes (Text.trim(Cin.get), " \t"))
                 Cout ("<") (element) ("> ");
        
        Cout.newline;
}
