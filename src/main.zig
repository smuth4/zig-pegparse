const std = @import("std");
const Allocator = std.mem.Allocator;
const zig_pegparse = @import("zig_pegparse");
//const regex = @import("regex");

const regex = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

const NodeList = std.ArrayList(Node);
const Node = struct {
    name: []const u8,
    start: usize,
    end: usize,
    children: ?NodeList = null, // null here means a leaf node

    fn print(self: *const Node, data: []const u8, i: u32) void {
        //if (self.name[0] == '_') {
        //    return;
        //}
        indent(i);
        std.debug.print("node name={s} start={d} end={d}\n", .{ self.name, self.start, self.end });
        indent(i);
        std.debug.print("value={s}\n", .{data[self.start..self.end]});
        if (self.children) |children| {
            for (children.items) |c| {
                //if (c.name[0] == '_') {
                //    continue;
                //}
                c.print(data, i + 2);
            }
        }
    }
};

/// Compiles a regex pattern string and returns a pattern code you can use
/// to match subjects. Returns `null` if something is wrong with the pattern
fn compile(needle: []const u8) ?*regex.pcre2_code_8 {
    const pattern: regex.PCRE2_SPTR8 = &needle[0];
    const patternLen: regex.PCRE2_SIZE = needle.len;
    var errornumber: c_int = undefined;
    var erroroffset: regex.PCRE2_SIZE = undefined;

    const regexp: ?*regex.pcre2_code_8 = regex.pcre2_compile_8(pattern, patternLen, 0, &errornumber, &erroroffset, null);
    if (regexp == null) {
        std.debug.print("re err: {d}\n", .{errornumber});
        var errbuf: [512]u8 = undefined;
        const buf: *regex.PCRE2_UCHAR8 = &errbuf[0];
        const writtenSize = regex.pcre2_get_error_message_8(errornumber, buf, 512);
        std.debug.print("re errmsg: {s}\n", .{errbuf[0..@intCast(writtenSize)]});
    }
    return regexp;
}

/// Takes in a compiled regexp pattern from `compile` and a string of
/// test which is the haystack and returns either the length of the
/// left-anchored match if found, or null if no match was found.
fn find(regexp: *regex.pcre2_code_8, haystack: []const u8) ?usize {
    if (haystack.len == 0) {
        return null;
    }
    const subject: regex.PCRE2_SPTR8 = &haystack[0];
    const subjLen: regex.PCRE2_SIZE = haystack.len;

    const matchData: ?*regex.pcre2_match_data_8 = regex.pcre2_match_data_create_from_pattern_8(regexp, null);
    // regex.PCRE2_ANCHORED prevents us from using JIT compilation, maybe it can be removed somehow?
    const rc: c_int = regex.pcre2_match_8(regexp, subject, subjLen, 0, regex.PCRE2_ANCHORED, matchData.?, null);

    if (rc < 0) {
        return null;
    }

    const ovector = regex.pcre2_get_ovector_pointer_8(matchData);
    // TODO: What is this actually checking? rc is set before ovector
    if (rc == 0) {
        std.debug.print("ovector was not big enough for all the captured substrings\n", .{});
        return null;
    }

    if (ovector[0] > ovector[1]) {
        std.debug.print("error with ovector\n", .{});
        regex.pcre2_match_data_free_8(matchData);
        regex.pcre2_code_free_8(regexp);
        return null;
    }
    return ovector[1] - ovector[0];
}

const ExpressionList = std.ArrayList(Expression);
const Literal = struct {
    name: []const u8,
    value: []const u8,
};

const Regex = struct {
    name: []const u8,
    value: []const u8,
    re: *regex.pcre2_code_8,
};

const Sequence = struct {
    name: []const u8,
    children: []const Expression,
};

const Choice = struct {
    name: []const u8,
    children: []const Expression,
};

// Handles ?, *, +, and {min,max}
const Quantity = struct {
    name: []const u8,
    min: usize,
    max: usize,
    child: *const Expression,
};

// Handles both ! and &
const Lookahead = struct {
    name: []const u8,
    negative: bool = false,
    child: *const Expression,
};

const GrammarError = error{ InvalidRegex, DuplicateLabel };

fn indent(level: u32) void {
    for (0..level) |_| {
        std.debug.print(" ", .{});
    }
}

const Expression = union(enum) {
    regex: Regex,
    literal: Literal,
    sequence: Sequence,
    quantity: Quantity,
`    choice: Choice,
    lookahead: Lookahead,

    fn print(self: *const Expression, i: u32) void {
        indent(i);
        switch (self.*) {
            .regex => |r| {
                std.debug.print("regex name={s}\n", .{r.name});
            },
            .literal => |l| {
                std.debug.print("literal name={s} value={s}\n", .{ l.name, l.value });
            },
            .sequence => |s| {
                std.debug.print("seq name={s}\n", .{s.name});
                for (s.children) |c| {
                    c.print(i + 2);
                }
            },
            .choice => |s| {
                std.debug.print("choice name={s}\n", .{s.name});
                for (s.children) |c| {
                    c.print(i + 2);
                }
            },
            .quantity => |q| {
                std.debug.print("quantity name={s} min={d} max={d}\n", .{ q.name, q.min, q.max });
                q.child.print(i + 2);
            },
            .lookahead => |l| {
                if (l.negative) {
                    std.debug.print("not name={s}\n", .{l.name});
                } else {
                    std.debug.print("lookahead name={s}\n", .{l.name});
                }
                l.child.print(i + 2);
            },
        }
    }

    // Return a tree of Nodes after parsing. Optionals are used to indicate if no match was found.
    fn parse(self: *const Expression, allocator: Allocator, data: []const u8, pos: *usize) !?Node {
        const toParse = data[pos.*..];
        std.debug.print("remaining: {s}\n", .{toParse});
        switch (self.*) {
            .regex => |r| {
                std.debug.print("regex name={s}\n", .{r.name});
                if (find(r.re, toParse)) |result| {
                    std.debug.print("regex match: {s}\n", .{toParse[0..result]});
                    const old_pos = pos.*;
                    pos.* += result;
                    return Node{ .name = r.name, .start = old_pos, .end = pos.* };
                } else {
                    return null;
                }
            },
            .literal => |l| {
                std.debug.print("literal value={s}\n", .{l.value});
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    return Node{ .name = l.name, .start = old_pos, .end = pos.* };
                }
            },
            .sequence => |s| {
                var children = std.ArrayList(Node).init(allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try c.parse(allocator, data, pos)) |n| {
                        try children.append(n);
                    } else {
                        pos.* = old_pos;
                        return null;
                    }
                }
                return Node{ .name = s.name, .start = old_pos, .end = pos.*, .children = children };
            },
            .choice => |s| {
                var children = std.ArrayList(Node).init(allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try c.parse(allocator, data, pos)) |n| {
                        try children.append(n);
                        return Node{ .name = s.name, .start = old_pos, .end = pos.*, .children = children };
                    } else {
                        pos.* = old_pos;
                    }
                }
                return null;
            },
            .lookahead => |l| {
                const old_pos = pos.*;
                const parsedNode = try l.child.parse(allocator, data, pos);
                pos.* = old_pos; // Always roll back the position
                if (parsedNode) |_| {
                    return if (l.negative) null else Node{ .name = l.name, .start = old_pos, .end = old_pos };
                } else {
                    return if (!l.negative) null else Node{ .name = l.name, .start = old_pos, .end = old_pos };
                }
            },
            .quantity => |q| {
                var children = std.ArrayList(Node).init(allocator);
                const old_pos = pos.*;
                for (0..q.max) |i| {
                    if (try q.child.parse(allocator, data, pos)) |parsedNode| {
                        try children.append(parsedNode);
                    } else {
                        if (i >= q.min and i <= q.max) {
                            return Node{ .name = q.name, .start = old_pos, .end = pos.*, .children = children };
                        } else {
                            return null;
                        }
                    }
                }
            },
        }
        return null;
    }
};

const ReferenceTable = std.StringHashMap(Expression);
const Grammar = struct {
    root: Expression,
    references: ReferenceTable,
    allocator: Allocator,

    pub fn init(allocator: Allocator) Grammar {
        return Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = Expression{ .literal = Literal{ .name = "_", .value = "" } },
            .allocator = allocator,
            .references = ReferenceTable.init(allocator),
        };
    }

    // Parse pre-loaded data according to the grammar
    pub fn parse(self: *Grammar, data: []const u8) !?Node {
        var pos: usize = 0;
        const n = try self.parseInner(&self.root, data, &pos);
        if (pos != data.len) {
            std.debug.print("did not reach end\n", .{});
            return null;
        }
        return n;
    }

    // Inside Grammar to have access to the reference table, but
    // ideally a reference for any visitor implementations
    const BootStrapVisitor = struct {
        const BootStrapVisitorSignature = *const fn (self: *BootStrapVisitor, data: []const u8, node: *const Node) anyerror!?Expression;
        visitorTable: std.StringHashMap(BootStrapVisitorSignature),
        allocator: Allocator,

        fn visit(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?Expression {
            try self.visitorTable.put("regex", &BootStrapVisitor.visit_regex);
            try self.visitorTable.put("rule", &BootStrapVisitor.visit_rule);
            try self.visitorTable.put("label_regex", &BootStrapVisitor.visit_label_regex);
            std.debug.assert(std.mem.eql(u8, node.name, "rules"));
            var rules = ExpressionList.init(self.allocator);
            for (node.children.?.items) |child| {
                if (try self.visit_generic(data, &child)) |result| {
                    std.debug.print("full rule", .{});
                    try rules.append(result);
                }
            }
            return self.visit_generic(data, node);
        }

        fn visit_generic(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?Expression {
            // Skip over anything starting with _ as convention
            if (node.name[0] == '_') {
                return null;
            }
            std.debug.print("visiting {s}\n", .{node.name});
            if (self.visitorTable.get(node.name)) |func| {
                return func(self, data, node);
            } else {
                if (node.children) |children| {
                    // Return the first non-null result by default
                    for (children.items) |child| {
                        if (try self.visit_generic(data, &child)) |result| {
                            return result;
                        }
                    }
                    return null;
                }
            }
            return null;
        }

        fn visit_rule(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?Expression {
            //const rule = self.visit_generic(data, node.children[2], expr);
            std.debug.print("visiting rule {s}\n", .{node.name});
            if (node.children) |children| {
                const label = try self.visit_generic(data, &children.items[0]);
                // We could descend and confirm the middle node is '=', but why bother
                const expression = try self.visit_generic(data, &children.items[2]);
                switch (label.?) {
                    .literal => |l| {
                        std.debug.print("visiting rule {s}\n", .{l.value});
                    },
                    else => unreachable,
                }
                return expression;
            }
            return null;
        }

        fn visit_label_regex(_: *BootStrapVisitor, data: []const u8, node: *const Node) !?Expression {
            // Send back an empty-named literal with the content as the value
            return createLiteral("", data[node.start..node.end]);
        }

        fn visit_regex(_: *BootStrapVisitor, _: []const u8, _: *const Node) !?Expression {
            const re = try createRegex("test", "test");
            return re;
        }
    };

    pub fn bootstrap(self: *Grammar, data: []const u8) !Expression {
        const n = try self.parse(data);
        var visitor = Grammar.BootStrapVisitor{
            .allocator = self.allocator,
            .visitorTable = std.StringHashMap(Grammar.BootStrapVisitor.BootStrapVisitorSignature).init(self.allocator),
        };
        //var root_expr = Expression{ .literal = Literal{ .name = "_", .value = "" } };

        const root_expr = try visitor.visit(data, &n.?);
        return root_expr.?;
    }

    fn addExpression(self: *Grammar, name: []const u8, expr: *Expression) !void {
        const result = try self.references.getOrPut(name);
        if (result.found_existing) {
            return GrammarError.DuplicateLabel;
        } else {
            result.value_ptr.* = expr;
        }
    }

    // Return a tree of Nodes after parsing. Optionals are used to indicate if no match was found.
    fn parseInner(self: *Grammar, exp: *const Expression, data: []const u8, pos: *usize) !?Node {
        const toParse = data[pos.*..];
        std.debug.print("remaining: {s}\n", .{toParse});
        switch (exp.*) {
            .regex => |r| {
                std.debug.print("regex name={s}\n", .{r.name});
                if (find(r.re, toParse)) |result| {
                    std.debug.print("regex match: {s}\n", .{toParse[0..result]});
                    const old_pos = pos.*;
                    pos.* += result;
                    return Node{ .name = r.name, .start = old_pos, .end = pos.* };
                } else {
                    return null;
                }
            },
            .literal => |l| {
                std.debug.print("literal value={s}\n", .{l.value});
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    return Node{ .name = l.name, .start = old_pos, .end = pos.* };
                }
            },
            .sequence => |s| {
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try self.parseInner(&c, data, pos)) |n| {
                        try children.append(n);
                    } else {
                        pos.* = old_pos;
                        return null;
                    }
                }
                return Node{ .name = s.name, .start = old_pos, .end = pos.*, .children = children };
            },
            .choice => |s| {
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try self.parseInner(&c, data, pos)) |n| {
                        try children.append(n);
                        return Node{ .name = s.name, .start = old_pos, .end = pos.*, .children = children };
                    } else {
                        pos.* = old_pos;
                    }
                }
                return null;
            },
            .lookahead => |l| {
                const old_pos = pos.*;
                const parsedNode = try self.parseInner(l.child, data, pos);
                pos.* = old_pos; // Always roll back the position
                if (parsedNode) |_| {
                    return if (l.negative) null else Node{ .name = l.name, .start = old_pos, .end = old_pos };
                } else {
                    return if (!l.negative) null else Node{ .name = l.name, .start = old_pos, .end = old_pos };
                }
            },
            .quantity => |q| {
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (0..q.max) |i| {
                    if (try self.parseInner(q.child, data, pos)) |parsedNode| {
                        try children.append(parsedNode);
                    } else {
                        if (i >= q.min and i <= q.max) {
                            return Node{ .name = q.name, .start = old_pos, .end = pos.*, .children = children };
                        } else {
                            return null;
                        }
                    }
                }
            },
        }
        return null;
    }
};

// Reduce boilerplate when manually constructing a grammar
fn createQuantity(name: []const u8, min: usize, max: usize, child: *const Expression) Expression {
    return Expression{ .quantity = Quantity{ .name = name, .min = min, .max = max, .child = child } };
}

fn createZeroOrMore(name: []const u8, child: *const Expression) Expression {
    return createQuantity(name, 0, std.math.maxInt(usize), child);
}

fn createOneOrMore(name: []const u8, child: *const Expression) Expression {
    return createQuantity(name, 1, std.math.maxInt(usize), child);
}

fn createChoice(name: []const u8, children: []const Expression) Expression {
    return Expression{ .choice = Choice{ .name = name, .children = children } };
}

fn createLookahead(name: []const u8, child: *const Expression) Expression {
    return Expression{ .lookahead = Lookahead{ .name = name, .negative = false, .child = child } };
}

fn createNot(name: []const u8, child: *const Expression) Expression {
    return Expression{ .lookahead = Lookahead{ .name = name, .negative = true, .child = child } };
}

fn createSequence(name: []const u8, children: []const Expression) Expression {
    return Expression{ .sequence = Sequence{ .name = name, .children = children } };
}

fn createRegex(name: []const u8, value: []const u8) !Expression {
    const re = compile(value) orelse return GrammarError.InvalidRegex;
    return Expression{ .regex = Regex{ .name = name, .value = value, .re = re } };
}

fn createLiteral(name: []const u8, value: []const u8) Expression {
    return Expression{ .literal = Literal{ .name = name, .value = value } };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const gallocator = gpa.allocator();
    var arena = std.heap.ArenaAllocator.init(gallocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Build a basic grammar for bootstrapping
    const ws = try createRegex("ws", "\\s+");
    const comment = try createRegex("comment", "#[^\r\n]*");
    const ignore = createZeroOrMore("_ignore", &createChoice(
        "_ignore",
        &[_]Expression{ ws, comment },
    ));
    const equals = createSequence("equals", &[_]Expression{ createLiteral("equal_literal", "="), ignore });
    const label = createSequence(
        "label",
        &[_]Expression{
            try createRegex("label_regex", "[a-zA-Z_][a-zA-Z_0-9]*"),
            ignore,
        },
    );
    const quoted_literal = createSequence(
        "quoted_literal",
        &[_]Expression{
            try createRegex("quoted_regex", "\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\""),
            ignore,
        },
    );
    const regex_exp = createSequence("regex", &[_]Expression{ createLiteral("tilde", "~"), quoted_literal });
    const reference = createSequence("reference", &[_]Expression{ label, createNot("_", &equals) });
    const atom = createChoice("atom", &[_]Expression{
        reference,
        quoted_literal,
        regex_exp,
    });
    const quantifier = createSequence("quantifier", &[_]Expression{ try createRegex("quantifier_re", "[*+?]"), ignore });
    const quantified = createSequence("quantified", &[_]Expression{ atom, quantifier });
    const term = createChoice("term", &[_]Expression{ atom, quantified });
    const term_plus = createOneOrMore("term_plus", &term); // deviation from parsimonious
    const sequence = createSequence("sequence", &[_]Expression{ term, term_plus });
    const or_term = createSequence("or_term", &[_]Expression{
        createLiteral("sep", "/"),
        ignore,
        term_plus,
    });
    const ored = createSequence("ored", &[_]Expression{
        term_plus,
        ignore,
        createOneOrMore("or_term_plus", &or_term),
    });
    const expression = createChoice("expression", &[_]Expression{ ored, sequence, term });
    const rule = createSequence("rule", &[_]Expression{
        label,
        equals,
        expression,
    });
    const rules = createSequence("rules", &[_]Expression{ ignore, createZeroOrMore("rules", &rule) });
    rule.print(0);

    var g = Grammar.init(allocator);
    g.root = rules;
    const testStr =
        \\test = "test" # comment
        \\test2 = reference
        \\meaninglessness = ~"\s+" / comment
        \\comment = ~"#[^\r\n]*"
    ;
    const n = try g.parse(testStr) orelse unreachable;
    n.print(testStr, 0);
    const full_expr = try g.bootstrap(testStr);
    std.debug.print("bootstrapped\n", .{});
    full_expr.print(0);
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
