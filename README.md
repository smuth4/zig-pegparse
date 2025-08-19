# zig-pegparse

zig-pegparse is an arbitrary-lookahead parser based on
[parsimonious](https://github.com/erikrose/parsimonious). It is able
to rapidly build parsers from a human-readable grammar definition, and
then use those parsers to build a abstract syntax tree.

## Install

Currently zig-pegparse requires
[pcre2](https://github.com/PCRE2Project/pcre2) for regexes, and as
such uses the latest version of zig it allows (currently 0.14.1).

```
zig fetch https://github.com/smuth4/zig-pegparse/archive/<commit>.tar.gz --save-exact=zig-pegparse
```

## Usage

```zig
const pegparse = @import("zig_pegparse");

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    grammar_factory = pegparse.Grammar.createFactory(allocator);
    defer grammar_factory.deinit();
    
    var grammar = grammar_factory.createGrammar(
        \\bold_text  = bold_open text bold_close
        \\text       = ~"[A-Z 0-9]*"i
        \\bold_open  = "(("
        \\bold_close = "))"
    );
    
    const data = "((bold stuff))";
    // Note that parsing the data does not create a copy of it.
    // If the data dissappears, the nodes are still technically traverseable, but reference data in invalid memory.
    var tree = grammar.parse(data);
    grammar.print(tree, data);
}
```

### Grammar Refrence



|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"string literal"` | A simple string literal, with support for escape sequences such as `\n`.                                                                             |
| `'string literal'` | Another form of string literals. Unlike double quotes, the only supported escape sequences are `\'` and `\\`                                         |
| `a`                | An example of a reference to another rule. Recursivity in references in support by PEG grammars, but must be carefully used to avoid infinite loops. |
| `a b c`            | A sequence, where all the items must match in that order.                                                                                            |
| `a / b / c`        | A choice, where only of one the items will be matched. Items will be tested in order until one succeeds or all of the fail                           |
| `~"\s+"i`          | Regex, which an optional flag at the end (`i` for case-insensitivity in this case)                                                                   |
| `(` `)`            | Used to group expression to ensure a certain priority, has no effect on actual parsing                                                               |


## Features

* Supported expr

## Goals

* Increase error handling facilities
  * Add optional diagnostic handler
* Reduce memory usage
  * Allow ignoring nodes with a specific prefix
  * Add cut operator
* Increase performance
  * Add packrat cache - currently a little difficult since we clear
    subexpressions for failed matches from the tree entirely, maybe
    keep them all then clone at the end?


## Limitations

- PEG parsers pretty much require the entire text to be available in memory. zig-pegparse does allow parsing partial strings, but in practice this is rarely useful since additional text may cause an entirely different parsing branch to be taken. The exception to this pattern is text such as YAML or JSONL, where individual entries have clearly defined barriers (although in those examples you'd be better served by a dedicated parser).
- Zig's string handling is basic: fast (good!) but unable to handle complex cases such as unicode (bad!). zig-pegparse leans heavily on pcre2 for handling these special cases, but it still introduces slowness.

## References

Bryan Ford's site is an incredible resource for anyone looking at PEG/packrat parsing: https://bford.info/packrat

- Foundational paper: https://bford.info/pub/lang/peg.pdf 
  - Paper on reducing memory usage with auto-inserted cut operators: https://kmizu.github.io/papers/paste513-mizushima.pdf
- Interesting PEG implementations:
  - https://github.com/erikrose/parsimonious - Python, clean API and error reporting
  - https://github.com/tolmasky/language - Objective-J, innovative in-tree parser error reporting (that may not translate well to Zig)
  - https://github.com/azatoth/PanPG - JS, takes a grammar and produces JS that can parse text
