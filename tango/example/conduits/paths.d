/*****************************************************

        How to create a path with all ancestors

*****************************************************/

import tango.io.FilePath;

void main (char[][] args)
{
        auto path = new FilePath (r"d/tango/foo/bar/wumpus");
        assert (path.create.exists && path.isFolder);
}
