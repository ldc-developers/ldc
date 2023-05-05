//===-- tools/timetrace2txt.d -------------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Converts --ftime-trace output to a text file.
//
//===----------------------------------------------------------------------===//

import core.stdc.stdlib : exit;
import std.stdio;
import std.file;
import std.json;
import std.range;
import std.conv;
import std.algorithm;


struct Config {
    string input_filename;
    string output_filename = "timetrace.txt";
    string output_TSV_filename;
    bool indented = false;
}
Config config;

File outputTextFile;
static string duration_format_string = "%13.3f ";

JSONValue sourceFile;
JSONValue[] metadata; // "M"
JSONValue[] counterdata; // "C"
JSONValue[] processes; // "X"

ulong lineNumberCounter = 1;

void parseCommandLine(string[] args) {
    import std.getopt : arraySep, getopt, defaultGetoptPrinter;

    try {
        arraySep = ";";
        auto helpInformation = getopt(
            args,
            "indent", "Output items as simply indented list instead of the fancy tree. This should go well with code-folding editors.", &config.indented,
            "o",   "Output filename (default: '" ~ config.output_filename ~ "'). Specify '-' to redirect output to stdout.", &config.output_filename,
            "tsv", "Also output to this file in duration-sorted Tab-Separated Values (TSV) format", &config.output_TSV_filename,
        );

        if (args.length != 2) {
            helpInformation.helpWanted = true;
            writeln("No input file given!\n");
        } else {
            config.input_filename = args[1];
            if (!exists(config.input_filename) || !isFile(config.input_filename)) {
                writefln("Input file '%s' does not exist or is not a file.\n", config.input_filename);
                helpInformation.helpWanted = true;
            }
        }

        if (helpInformation.helpWanted) {
            defaultGetoptPrinter(
                "Converts --ftime-trace output to text.\n" ~
                "Usage: timetrace2txt [input file] [options]\n",
                helpInformation.options
            );
            exit(1);
        }

    }
    catch (Exception e) {
        writefln("Error processing command line arguments: %s", e.msg);
        writeln("Use '--help' for help.");
        exit(1);
    }
}

int main(string[] args)
{
    parseCommandLine(args);

    outputTextFile = (config.output_filename == "-") ? stdout : File(config.output_filename, "w");

    auto input_json = read(config.input_filename).to!string;
    sourceFile = parseJSON(input_json);
    processInputJSON();
    constructTree();
    constructList();

    {
        outputTextFile.writeln("Timetrace: ", args[1]);
        lineNumberCounter++;

        outputTextFile.writeln("Metadata:");
        lineNumberCounter++;

        foreach (node; metadata)
        {
            outputTextFile.write("  ");
            outputTextFile.writeln(node);
            lineNumberCounter++;
        }

        outputTextFile.writeln("Duration (ms)");
        lineNumberCounter++;
    }

    wchar[] indentstring;
    foreach (i, ref child; Node.root.children) {
        if (config.indented)
            child.printIndented(indentstring);
        else
            child.printTree(indentstring);
    }

    if (config.output_TSV_filename.length != 0) {
        File outputTSVFile = (config.output_TSV_filename == "-") ? stdout : File(config.output_TSV_filename, "w");
        outputTSVFile.writeln("Duration\tText Line Number\tName\tLocation\tDetail");
        foreach (node; Node.all)
            outputTSVFile.writeln(node.duration, "\t", node.lineNumber, "\t",
                    node.name, "\t", node.location, "\t", node.detail);
    }

    return 0;
}

void processInputJSON()
{
    auto beginningOfTime = sourceFile["beginningOfTime"].integer;
    auto traceEvents = sourceFile["traceEvents"].array;

    // Read meta data
    foreach (value; traceEvents)
    {
        switch (value["ph"].str)
        {
        case "M":
            metadata ~= value;
            break;
        case "C":
            counterdata ~= value;
            break;
        case "X":
            processes ~= value;
            break;
        default: //drop
        }
    }

    // process node = {"ph":"X","name": "Sema1: Module object","ts":26825,"dur":1477,"loc":"<no file>","args":{"detail": "","loc":"<no file>"},"pid":101,"tid":101},
    // Sort time processes
    multiSort!(q{a["ts"].integer < b["ts"].integer}, q{a["dur"].integer > b["dur"].integer})(processes);
}

// Build tree (to get nicer looking structure lines)
void constructTree()
{
    Node.root = Node(new JSONValue("Tree root"), long.max, true);
    Node.count++;
    Node*[] parent_stack = [&Node.root]; // each stack item represents the first uncompleted note of that level in the tree

    foreach (ref process; processes)
    {
        auto last_ts = process["ts"].integer + process["dur"].integer;
        size_t parent_idx = 0; // index in parent_stack to which this item should be added.

        foreach (i; 0 .. parent_stack.length)
        {
            if (last_ts > parent_stack[i].last_ts)
            {
                // The current process outlasts stack item i. Stop traversing, parent is i-1;
                parent_idx = i - 1;
                parent_stack.length = i;
                break;
            }

            parent_idx = i;
        }

        parent_stack[parent_idx].children ~= Node(&process, last_ts);
        parent_stack ~= &parent_stack[parent_idx].children[$ - 1];
        Node.count++;
    }
}

void constructList()
{
    size_t offset;

    Node.all.length = Node.count - 1;

    void handle(Node* root)
    {
        Node.all[offset++] = root;

        foreach (ref child; root.children)
            handle(&child);
    }

    foreach (ref child; Node.root.children)
        handle(&child);

    Node.all.sort!((a, b) => a.duration > b.duration);
}

struct Node
{
    Node[] children;
    JSONValue* json;
    long last_ts; // represents the last timestamp of this node (i.e. ts + dur)
    ulong lineNumber;

    string name;
    long duration; // microseconds
    string location;
    string detail;

    static Node root;
    static Node*[] all;
    static size_t count = 0;

    this(JSONValue* json, long last_ts, bool root = false)
    {
        this.json = json;
        this.last_ts = last_ts;

        if (!root)
        {
            this.duration = (*json)["dur"].integer;
            this.name = (*json)["name"].str;
            if (auto args = "args" in *json) {
                if (auto value = "loc" in *args) {
                    this.location = value.str;
                }
                if (auto value = "detail" in *args) {
                    this.detail = value.str;
                }
            }
        }
    }

    void printTree(wchar[] indentstring, bool last_child = false) {
        this.lineNumber = lineNumberCounter;
            lineNumberCounter++;

        // Output in milliseconds.
        outputTextFile.writef(duration_format_string, cast(double)(this.duration) / 1000);

        if (last_child)
            indentstring[$-1] = '└';
        outputTextFile.write(indentstring);
        outputTextFile.write("- ", this.name);
        outputTextFile.write(", ", this.detail);
        outputTextFile.writeln(", ", this.location);
        if (last_child)
            indentstring[$-1] = ' ';

        wchar[] child_indentstring = indentstring ~ " |";
        foreach (i, ref child; this.children) {
            child.printTree(child_indentstring, i == this.children.length-1);
        }
    }

    void printIndented(wchar[] indentstring) {
        this.lineNumber = lineNumberCounter;
            lineNumberCounter++;

        // Output in milliseconds.
        outputTextFile.write(indentstring);
        outputTextFile.write("- ");
        outputTextFile.writef(duration_format_string, cast(double)(this.duration) / 1000);
        outputTextFile.write("- ", this.name);
        outputTextFile.write(", ", this.detail);
        outputTextFile.writeln(", ", this.location);

        wchar[] child_indentstring = indentstring ~ "  ";
        foreach (i, ref child; this.children) {
            child.printIndented(child_indentstring);
        }
    }
}
