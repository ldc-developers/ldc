// RUN: %ldc -c -singleobj %s %S/inputs/gh2777b.d

int main(string[] args);

void reset() { main(null); }
