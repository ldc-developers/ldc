module FileBucket;

private import  tango.io.FilePath,
                tango.io.FileConduit;

private import  tango.core.Exception;

/******************************************************************************

        FileBucket implements a simple mechanism to store and recover a 
        large quantity of data for the duration of the hosting process.
        It is intended to act as a local-cache for a remote data-source, 
        or as a spillover area for large in-memory cache instances. 
        
        Note that any and all stored data is rendered invalid the moment
        a FileBucket object is garbage-collected.

        The implementation follows a fixed-capacity record scheme, where
        content can be rewritten in-place until said capacity is reached.
        At such time, the altered content is moved to a larger capacity
        record at end-of-file, and a hole remains at the prior location.
        These holes are not collected, since the lifespan of a FileBucket
        is limited to that of the host process.

        All index keys must be unique. Writing to the FileBucket with an
        existing key will overwrite any previous content. What follows
        is a contrived example:
        
        ---
        char[] text = "this is a test";

        auto bucket = new FileBucket (new FilePath("bucket.bin"), FileBucket.HalfK);

        // insert some data, and retrieve it again
        bucket.put ("a key", text);
        char[] b = cast(char[]) bucket.get ("a key");

        assert (b == text);
        bucket.close;
        ---

******************************************************************************/

class FileBucket
{
        /**********************************************************************

                Define the capacity (block-size) of each record

        **********************************************************************/

        struct BlockSize
        {
                int capacity;
        }

        // basic capacity for each record
        private FilePath                path;

        // basic capacity for each record
        private BlockSize               block;

        // where content is stored
        private FileConduit             file;

        // pointers to file records
        private Record[char[]]          map;

        // current file size
        private long                    fileSize;

        // current file usage
        private long                    waterLine;

        // supported block sizes
        public static const BlockSize   EighthK  = {128-1},
                                        HalfK    = {512-1},
                                        OneK     = {1024*1-1},
                                        TwoK     = {1024*2-1},
                                        FourK    = {1024*4-1},
                                        EightK   = {1024*8-1},
                                        SixteenK = {1024*16-1},
                                        ThirtyTwoK = {1024*32-1},
                                        SixtyFourK = {1024*64-1};


        /**********************************************************************

                Construct a FileBucket with the provided path and record-
                size. Selecting a record size that roughly matches the 
                serialized content will limit 'thrashing'.

        **********************************************************************/

        this (char[] path, BlockSize block)
        {
                this (new FilePath(path), block);
        }

        /**********************************************************************

                Construct a FileBucket with the provided path, record-size,
                and inital record count. The latter causes records to be 
                pre-allocated, saving a certain amount of growth activity.
                Selecting a record size that roughly matches the serialized 
                content will limit 'thrashing'. 

        **********************************************************************/

        this (FilePath path, BlockSize block, uint initialRecords = 100)
        {
                this.path = path;
                this.block = block;

                // open a storage file
                file = new FileConduit (path, FileConduit.ReadWriteCreate);

                // set initial file size (can be zero)
                fileSize = initialRecords * block.capacity;
                file.seek (fileSize);
                file.truncate ();
        }

        /**********************************************************************

                Return the block-size in use for this FileBucket

        **********************************************************************/

        int getBufferSize ()
        {
                return block.capacity+1;
        }

        /**********************************************************************
        
                Return where the FileBucket is located

        **********************************************************************/

        FilePath getFilePath ()
        {
                return path;
        }

        /**********************************************************************

                Return the currently populated size of this FileBucket

        **********************************************************************/

        synchronized long length ()
        {
                return waterLine;
        }

        /**********************************************************************

                Return the serialized data for the provided key. Returns
                null if the key was not found.

        **********************************************************************/

        synchronized void[] get (char[] key)
        {
                Record r = null;

                if (key in map)
                   {
                   r = map [key];
                   return r.read (this);
                   }
                return null;
        }

        /**********************************************************************

                Remove the provided key from this FileBucket.

        **********************************************************************/

        synchronized void remove (char[] key)
        {
                map.remove(key);
        }

        /**********************************************************************

                Write a serialized block of data, and associate it with
                the provided key. All keys must be unique, and it is the
                responsibility of the programmer to ensure this. Reusing 
                an existing key will overwrite previous data. 

                Note that data is allowed to grow within the occupied 
                bucket until it becomes larger than the allocated space.
                When this happens, the data is moved to a larger bucket
                at the file tail.

        **********************************************************************/

        synchronized void put (char[] key, void[] data)
        {
                Record* r = key in map;

                if (r is null)
                   {
                   auto rr = new Record;
                   map [key] =  rr;
                   r = &rr;
                   }
                r.write (this, data, block);
        }

        /**********************************************************************

                Close this FileBucket -- all content is lost.

        **********************************************************************/

        synchronized void close ()
        {
                if (file)
                   {
                   file.detach;
                   file = null;
                   map = null;
                   }
        }

        /**********************************************************************

                Each Record takes up a number of 'pages' within the file. 
                The size of these pages is determined by the BlockSize 
                provided during FileBucket construction. Additional space
                at the end of each block is potentially wasted, but enables 
                content to grow in size without creating a myriad of holes.

        **********************************************************************/

        private static class Record
        {
                private long            offset;
                private int             length,
                                        capacity = -1;

                /**************************************************************

                **************************************************************/

                private static void eof (FileBucket bucket)
                {
                        throw new IOException ("Unexpected EOF in FileBucket '"~bucket.path.toString()~"'");
                }

                /**************************************************************

                        This should be protected from thread-contention at
                        a higher level.

                **************************************************************/

                void[] read (FileBucket bucket)
                {
                        void[] data = new ubyte [length];

                        bucket.file.seek (offset);
                        if (bucket.file.read (data) != length)
                            eof (bucket);

                        return data;
                }

                /**************************************************************

                        This should be protected from thread-contention at
                        a higher level.

                **************************************************************/

                void write (FileBucket bucket, void[] data, BlockSize block)
                {
                        length = data.length;

                        // create new slot if we exceed capacity
                        if (length > capacity)
                            createBucket (bucket, length, block);

                        // locate to start of content 
                        bucket.file.seek (offset);
        
                        // write content
                        if (bucket.file.write (data) != length)
                            eof (bucket);
                }

                /**************************************************************

                **************************************************************/

                void createBucket (FileBucket bucket, int bytes, BlockSize block)
                {
                        offset = bucket.waterLine;
                        capacity = (bytes + block.capacity) & ~block.capacity;

                        bucket.waterLine += capacity;
                        if (bucket.waterLine > bucket.fileSize)
                           {
                           // grow the filesize 
                           bucket.fileSize = bucket.waterLine * 2;

                           // expand the physical file size
                           bucket.file.seek (bucket.fileSize);
                           bucket.file.truncate ();
                           }
                }
        }
}


