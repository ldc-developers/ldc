void foo() { void delegate()[] bar; try {} finally { foreach (dg; bar) dg(); } }
