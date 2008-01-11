module example.reference.chapter11;

import tango.util.collection.HashMap;
import tango.util.collection.ArrayBag;
import tango.util.collection.LinkSeq;
import tango.util.collection.CircularSeq;
import tango.util.collection.ArraySeq;
import tango.util.collection.TreeBag;
import tango.util.collection.iterator.FilteringIterator;
import tango.util.collection.iterator.InterleavingIterator;

import tango.io.Stdout;
import tango.core.Exception;

import tango.util.collection.model.Comparator;
import tango.util.collection.impl.BagCollection;
import tango.util.collection.impl.SeqCollection;
import Ascii = tango.text.Ascii;

void linkedListExample(){
    Stdout.format( "linkedListExample" ).newline;
    alias LinkSeq!(char[]) StrList;
    StrList lst = new StrList();

    lst.append( "value1" );
    lst.append( "value2" );
    lst.append( "value3" );

    auto it = lst.elements();
    // The call to .more gives true, if there are more elements
    // and switches the iterator to the next one if available.
    while( it.more ){
        char[] item_value = it.get;
        Stdout.format( "Value:{0}", item_value ).newline;
    }
}

void hashMapExample(){
    Stdout.format( "hashMapExample" ).newline;
    alias HashMap!(char[], char[]) StrStrMap;
    StrStrMap map = new StrStrMap();
    map.add( "key1", "value1" );
    char[] key = "key1";
    Stdout.format( "Key: {0}, Value:{1}", key, map.get( key )).newline;


    auto it = map.keys();
    // The call to .more gives true, if there are more elements
    // and switches the iterator to the next one if available.
    while( it.more ){
        char[] item_key = void;
        char[] item_value = it.get( item_key ); // only for maps, the key is returns via inout
        Stdout.format( "Key: {0}, Value:{1}", item_key, item_value ).newline;
    }
}

void testComparator(){
    char[][] result;

    // Create and fill the containers
    auto nameSet = new TreeBag!(char[])( null, new class() Comparator!(char[]){
        int compare( char[] first, char[] second ){
            return Ascii.icompare( first, second );
        }
    });

    nameSet.addIf( "Alice" );
    nameSet.addIf( "Bob" );
    nameSet.addIf( "aliCe" );

    // use foreach to iterate over the container
    foreach ( char[] i; nameSet )
        result ~= i;
 foreach( char[] i; result ) Stdout.format( "{0} ", i );
 Stdout.newline;

    // test the result
    assert( result == [ "Alice", "Bob" ], "testIterator" );
}

void testSreener(){
    int[] result;

    // Create and fill the containers
    auto ratioSamples = new ArraySeq!(float)( (float v){
        return v >= 0.0f && v < 1.0f;
    });

    ratioSamples.append( 0.0f );
    ratioSamples.append( 0.5f );
    ratioSamples.append( 0.99999f );
    // try to insert a value that is not allowed
    try{
        ratioSamples.append( 1.0f );
        // will never get here
        assert( false );
    } catch( IllegalElementException e ){
    }
}


void testForeach(){
    int[] result;

    // Create and fill the containers
    auto l1 = new CircularSeq!(int);
    for( int i = 0; i < 6; i+=2 ){
        l1.append( i );
    }

    // use foreach to iterate over the container
    foreach ( int i; l1 )
        result ~= i;
//     foreach_reverse ( int i; l1 )
//         result ~= i;

    // test the result
    assert( result == [ 0, 2, 4 ], "testIterator" );
}

void testIterator(){
    int[] result;

    // Create and fill the containers
    auto l1 = new LinkSeq!(int);
    for( int i = 0; i < 6; i+=2 ){
        l1.append( i );
    }

    // define the Iterator
    auto it = l1.elements();

    // use the Iterator to iterate over the container
    while (it.more())
        result ~= it.get();

    // test the result
    assert( result == [ 0, 2, 4 ], "testIterator" );
}

void testFilteringIterator(){
    int[] result;
    alias ArrayBag!(int) IntBag;

    // Create and fill the container
    auto ib = new IntBag;
    for( int i = 0; i < 20; i++ ){
        ib.add( i );
    }

    // define the FilteringIterator with a function literal
    auto it = new FilteringIterator!(int)( ib.elements(), (int i){
        return i >= 3 && i < 7;
    });

    // use the Iterator with the more()/get() pattern
    while (it.more())
        result ~= it.get();

    // test the result
    assert( result == [ 3, 4, 5, 6 ], "testFilteringIterator" );

}

void testFilteringIteratorRemove(){
    int[] result;

    // Create and fill the container
    auto container = new LinkSeq!(int);
    for( int i = 0; i < 10; i++ ){
        container.append( i );
    }

    // 1. Build a list of elements to delete
    auto dellist = new LinkSeq!(int);
    foreach( int i; container ){
        if( i < 3 || i >= 7 ){
            // container.remove( i ); /* NOT POSSIBLE */
            dellist.append( i );
        }
    }
    // 2. Iterate over this deletion list and
    // delete the items in the original container.
    foreach( int i; dellist ){
        container.remove( i );
    }

    foreach ( int i; container )
        result ~= i;

    // test the result
    assert( result == [ 3, 4, 5, 6 ], "testFilteringIterator" );

}

void testInterleavingIterator(){
    int[] result;

    // Create and fill the containers
    auto l1 = new LinkSeq!(int);
    auto l2 = new ArraySeq!(int);
    for( int i = 0; i < 6; i+=2 ){
        l1.append( i );
        l2.append( i+1 );
    }

    // define the InterleavingIterator
    auto it = new InterleavingIterator!(int)( l1.elements(), l2.elements() );

    // use the InterleavingIterator to iterate over the container
    while (it.more())
        result ~= it.get();

    // test the result
    assert( result == [ 0, 1, 2, 3, 4, 5 ], "testInterleavingIterator" );
}

// foreach( int i; result ) Stdout.format( "{0} ", i );
// Stdout.newline;

void main(){
    Stdout.format( "reference - Chapter 11 Example" ).newline;
    hashMapExample();
    linkedListExample();

    testSreener();
    testComparator();
    testForeach();
    testIterator();
    testFilteringIterator();
    testInterleavingIterator();
    testFilteringIteratorRemove();

    Stdout.format( "=== End ===" ).newline;
}


