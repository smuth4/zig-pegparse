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
const std = @import("std");
const pegparse = @import("zig_pegparse");

pub fn main() !void {
    // Feel free to use whatever allocator you want
    const allocator = std.heap.c_allocator;

    // The factory here is a just a specialized grammar that can create more grammars
    var grammar_factory = pegparse.Grammar.initFactory(allocator);
    defer grammar_factory.deinit();

    // Read in the PEG definition
    var grammar = grammar_factory.createGrammar(
        \\bold_text  = bold_open text bold_close
        \\text       = ~"[A-Z 0-9]*"i
        \\bold_open  = "(("
        \\bold_close = "))"
    );
    defer grammar.deinit();

    // Parse an example string
    const data = "((bold stuff))";
    // Note that parsing the data does not create a copy of it.
    // If the data dissappears, the nodes are still technically traverseable, but references invalid data offsets.
    var tree = grammar.parse(data);
    defer tree.deinit();

    // A tree is always returned, even if it didn't parse everything.
    if (tree.root().?.value.start != data.len) {
        std.debug.print("Parsing did not complete", .{});
    }
}
```

Once a tree is built, there are many options for traversing it. The simplest option is the built-in iterators:

```zig
var iter = tree.iterator(.{});
while (iter.next()) |e| {
    if (e.exp.name == "bold_open") {
        std.debug.print("<b>", .{});
    } else if (e.exp.name == "bold_close") {
        std.debug.print("</b>", .{});
    } else if (e.exp.name == "text") {
        const nodeData = data[e.start..e.end];
        std.debug.print("{s}", .{nodeData});
    } else {
        std.debug.print("Unknown node type \"{s}\"!", .{node.value.expr.*.name});
    }
}
```

If you need more control, you may want to consider the visitor
pattern. It will allow for more complicated decision making about when
and where to descend the tree. Since zig-pegparse dogfoods it's own
grammar logic, see `ExpressionVisitor` for a more complete example.

```zig
const NodeVisitor = struct {
    const NodeVisitorSignature = *const fn (self: *NodeVisitor, data: []const u8, node: *const Node) void;

    const visitor_table = std.static_string_map.StaticStringMap(ExpressionVisitorSignature).initComptime(.{
        .{ "bold_open", visit_bold_open },
        .{ "bold_close", visit_bold_close },
        .{ "bold_text", visit_bold_text },
    });

    fn visit_generic(self: *ExpressionVisitor, data: []const u8, node: *const Node) void {
            if (visitor_table.get(node.value.expr.*.name)) |func| {
                func(self, data, node);
            } else {
                std.debug.print("Unknown node type \"{s}\"!", .{node.value.expr.*.name});
            }
        }
    }

    fn visit_bold_open(_: *ExpressionVisitor, _: []const u8, _: *const Node) void {
        std.debug.print("<b>", .{});
    }
    fn visit_bold_close(_: *ExpressionVisitor, _: []const u8, _: *const Node) void {
        std.debug.print("</b>", .{});
    }
    fn visit_bold_close(_: *ExpressionVisitor, data: []const u8, node: *const Node) void {
        const nodeData = data[e.start..e.end];
        std.debug.print("{s}", .{nodeData});
    }
}

nv = NodeVisitor{};
nv.visit_generic(tree.root().?);

```

### Grammar Refrence

| Example            | Notes                                                                                                                                                |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"string literal"` | A simple string literal, with support for escape sequences such as `\n`.                                                                             |
| `'string literal'` | Another form of string literals. Unlike double quoted literals, the only supported escape sequences are `\'` and `\\`                                |
| `a`                | An example of a reference to another rule. Recursivity in references in support by PEG grammars, but must be carefully used to avoid infinite loops. |
| `a b c`            | A sequence, where all the items must match in that order.                                                                                            |
| `a / b / c`        | A choice, where only of one the items will be matched. Items will be tested in order until one succeeds or all of the fail                           |
| `~"\s+"i`          | Regex, with an optional flag at the end (`i` for case-insensitivity in this case). Escape sequences are passed directly to PCRE2.                    |
| `(` `)`            | Used to group expression to ensure a certain priority, has no effect on actual parsing                                                               |

## Performance

While a PEG parser will never beat out a dedicated state machine or the like, it should still be pretty darn fast. Parsimonious' section on [optimizing grammars](https://github.com/erikrose/parsimonious?tab=readme-ov-file#optimizing-grammars) is very relevant to zig-pegparser's grammars as well.

zig-pegparse makes use of a [packrat cache](https://en.wikipedia.org/wiki/Packrat_parser#Memoization_technique) to prevent excessive backtracking. However, there are grammars that don't do much backtracking in general, leading to both increased memory usage and adverse performance impact. In these situations, you can disable the cache:

```zig
grammar.disableCache();
```

## Goals

* Increase error handling facilities
  * Add optional diagnostic handler for parse errors
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
