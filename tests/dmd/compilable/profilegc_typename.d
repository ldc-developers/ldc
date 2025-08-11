// REQUIRED_ARGS: -profile=gc
// DISABLED: LDC // -profile=gc not supported
struct T(string s) {}
alias TypeWithQuotes = T!q"EOS
`"'}])>
EOS";

void foo() {
    TypeWithQuotes[] arr;
    arr ~= TypeWithQuotes();
}
