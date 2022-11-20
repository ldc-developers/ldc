module imports.ldc_github_739b;

import ldc_github_739;
import imports.ldc_github_739a;

struct MatchTree {
    struct TerminalTag {}
    TerminalTag[] m_terminalTags;

    void print() {
        m_terminalTags.map!(t => "").array;
    }
}
