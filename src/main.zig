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
    value: []const u8,
};

const Regex = struct {
    value: []const u8,
    re: *regex.pcre2_code_8,
};

const Sequence = struct {
    children: []const Expression,
};

const Choice = struct {
    children: []const Expression,
};

// Handles ?, *, +, and {min,max}
const Quantity = struct {
    min: usize,
    max: usize,
    child: *const Expression,
};

// Handles both ! and &
const Lookahead = struct {
    negative: bool = false,
    child: *const Expression,
};

const GrammarError = error{ InvalidRegex, DuplicateLabel };

// Utility function, make better later
fn indent(level: u32) void {
    for (0..level) |_| {
        std.debug.print(" ", .{});
    }
}

const Expression = struct {
    pub const Matcher = union(enum) {
        regex: Regex,
        literal: Literal,
        sequence: Sequence,
        quantity: Quantity,
        choice: Choice,
        lookahead: Lookahead,
    };

    name: []const u8,
    matcher: Matcher,

    fn print(self: *const Expression, i: u32) void {
        indent(i);
        switch (self.*.matcher) {
            .regex => |_| {
                std.debug.print("regex name={s}\n", .{self.name});
            },
            .literal => |l| {
                std.debug.print("literal name={s} value={s}\n", .{ self.name, l.value });
            },
            .sequence => |s| {
                std.debug.print("seq name={s}\n", .{self.name});
                for (s.children) |c| {
                    c.print(i + 2);
                }
            },
            .choice => |s| {
                std.debug.print("choice name={s}\n", .{self.name});
                for (s.children) |c| {
                    c.print(i + 2);
                }
            },
            .quantity => |q| {
                std.debug.print("quantity name={s} min={d} max={d}\n", .{ self.name, q.min, q.max });
                q.child.print(i + 2);
            },
            .lookahead => |l| {
                if (l.negative) {
                    std.debug.print("not name={s}\n", .{self.name});
                } else {
                    std.debug.print("lookahead name={s}\n", .{self.name});
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
                std.debug.print("regex name={s}\n", .{self.name});
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
                    return Node{ .name = self.name, .start = old_pos, .end = pos.* };
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
                return Node{ .name = self.name, .start = old_pos, .end = pos.*, .children = children };
            },
            .choice => |s| {
                var children = std.ArrayList(Node).init(allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try c.parse(allocator, data, pos)) |n| {
                        try children.append(n);
                        return Node{ .name = self.name, .start = old_pos, .end = pos.*, .children = children };
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

const ReferenceTable = std.StringHashMap(*Expression);
const Grammar = struct {
    // Where parsing starts from
    root: Expression,
    // Holds points to expressions for reference lookups
    references: ReferenceTable,
    allocator: Allocator,

    pub fn init(allocator: Allocator) Grammar {
        return Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = Expression{ .name = "", .matcher = .{ .literal = .{ .value = "" } } },
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
        grammar: *Grammar,

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
            // Skip over anything without a name or a name starting with '_'
            if (node.name.len == 0 or node.name[0] == '_') {
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
                switch (label.?.matcher) {
                    .literal => |l| {
                        std.debug.print("visiting rule {s}\n", .{l.value});
                    },
                    else => unreachable,
                }
                return expression;
            }
            return null;
        }

        fn visit_label_regex(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?Expression {
            // Send back an empty-named literal with the content as the value
            return try self.grammar.createLiteral("", data[node.start..node.end]);
        }

        fn visit_regex(self: *BootStrapVisitor, _: []const u8, _: *const Node) !?Expression {
            const re = try self.grammar.createRegex("test", "test");
            return re;
        }
    };

    fn addExpression(self: *Grammar, name: []const u8, expr: *Expression) !void {
        // Expressions with empty names can only be referenced directly
        if (name.len == 0 or name[0] == '_') {
            return;
        }
        std.debug.print("insert: {s}\n", .{name});
        const result = try self.references.getOrPut(name);
        if (result.found_existing) {
            std.debug.print("dupe: {s}\n", .{name});
            return GrammarError.DuplicateLabel;
        } else {
            result.value_ptr.* = expr;
        }
    }

    // Reduce boilerplate when manually constructing a grammar
    fn createQuantity(self: *Grammar, name: []const u8, min: usize, max: usize, child: *const Expression) !Expression {
        var expr = Expression{ .name = name, .matcher = .{ .quantity = Quantity{ .min = min, .max = max, .child = child } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createZeroOrMore(self: *Grammar, name: []const u8, child: *const Expression) !Expression {
        return self.createQuantity(name, 0, std.math.maxInt(usize), child);
    }

    fn createOneOrMore(self: *Grammar, name: []const u8, child: *const Expression) !Expression {
        return self.createQuantity(name, 1, std.math.maxInt(usize), child);
    }

    fn createChoice(self: *Grammar, name: []const u8, children: []const Expression) !Expression {
        var expr = Expression{ .name = name, .matcher = .{ .choice = Choice{ .children = children } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createLookahead(self: *Grammar, name: []const u8, child: *const Expression) !Expression {
        var expr = Expression{ .name = name, .matcher = .{ .lookahead = Lookahead{ .negative = false, .child = child } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createNot(self: *Grammar, name: []const u8, child: *const Expression) !Expression {
        var expr = Expression{ .name = name, .matcher = .{ .lookahead = Lookahead{ .negative = true, .child = child } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createSequence(self: *Grammar, name: []const u8, children: []const Expression) !Expression {
        var expr = Expression{ .name = name, .matcher = .{ .sequence = Sequence{ .children = children } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createRegex(self: *Grammar, name: []const u8, value: []const u8) !Expression {
        const re = compile(value) orelse return GrammarError.InvalidRegex;
        var expr = Expression{ .name = name, .matcher = .{ .regex = Regex{ .value = value, .re = re } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    fn createLiteral(self: *Grammar, name: []const u8, value: []const u8) !Expression {
        var expr = Expression{ .name = name, .matcher = Expression.Matcher{ .literal = Literal{ .value = value } } };
        try self.addExpression(expr.name, &expr);
        return expr;
    }

    pub fn bootstrap(self: *Grammar) !Expression {
        // Build a basic grammar for bootstrapping
        const ws = try self.createRegex("ws", "\\s+");
        const comment = try self.createRegex("comment", "#[^\r\n]*");
        const ignore = try self.createZeroOrMore("_ignore", &try self.createChoice(
            "",
            &[_]Expression{ ws, comment },
        ));
        const equals = try self.createSequence("equals", &[_]Expression{ try self.createLiteral("", "="), ignore });
        const label = try self.createSequence(
            "label",
            &[_]Expression{
                try self.createRegex("label_regex", "[a-zA-Z_][a-zA-Z_0-9]*"),
                ignore,
            },
        );
        const quoted_literal = try self.createSequence(
            "quoted_literal",
            &[_]Expression{
                try self.createRegex("quoted_regex", "\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\""),
                ignore,
            },
        );
        const regex_exp = try self.createSequence("regex", &[_]Expression{ try self.createLiteral("", "~"), quoted_literal });
        const reference = try self.createSequence("reference", &[_]Expression{ label, try self.createNot("_", &equals) });
        const atom = try self.createChoice("atom", &[_]Expression{
            reference,
            quoted_literal,
            regex_exp,
        });
        const quantifier = try self.createSequence("quantifier", &[_]Expression{ try self.createRegex("", "[*+?]"), ignore });
        const quantified = try self.createSequence("quantified", &[_]Expression{ atom, quantifier });
        const term = try self.createChoice("term", &[_]Expression{ atom, quantified });
        const term_plus = try self.createOneOrMore("term_plus", &term); // deviation from parsimonious
        const sequence = try self.createSequence("sequence", &[_]Expression{ term, term_plus });
        const or_term = try self.createSequence("or_term", &[_]Expression{
            try self.createLiteral("", "/"),
            ignore,
            term_plus,
        });
        const ored = try self.createSequence("ored", &[_]Expression{
            term_plus,
            ignore,
            try self.createOneOrMore("or_term_plus", &or_term),
        });
        const expression = try self.createChoice("expression", &[_]Expression{ ored, sequence, term });
        const rule = try self.createSequence("rule", &[_]Expression{
            label,
            equals,
            expression,
        });
        const rules = try self.createSequence("rules", &[_]Expression{ ignore, try self.createZeroOrMore("", &rule) });

        // Assign the rules to ourselves
        self.root = rules;
        const rule_data =
            \\test = "test" # comment
            \\test2 = reference
            \\meaninglessness = ~"\s+" / comment
            \\comment = ~"#[^\r\n]*"
        ;
        const n = try self.parse(rule_data);
        var visitor = Grammar.BootStrapVisitor{
            .grammar = self,
            .allocator = self.allocator,
            .visitorTable = std.StringHashMap(Grammar.BootStrapVisitor.BootStrapVisitorSignature).init(self.allocator),
        };

        // Bootstrap against the full rules
        const root_expr = try visitor.visit(rule_data, &n.?);
        return root_expr.?;
    }

    // Return a tree of Nodes after parsing. Optionals are used to indicate if no match was found.
    fn parseInner(self: *Grammar, exp: *const Expression, data: []const u8, pos: *usize) !?Node {
        const toParse = data[pos.*..];
        std.debug.print("remaining: {s}\n", .{toParse});
        switch (exp.*.matcher) {
            .regex => |r| {
                std.debug.print("regex name={s}\n", .{exp.name});
                if (find(r.re, toParse)) |result| {
                    std.debug.print("regex match: {s}\n", .{toParse[0..result]});
                    const old_pos = pos.*;
                    pos.* += result;
                    return Node{ .name = exp.name, .start = old_pos, .end = pos.* };
                } else {
                    return null;
                }
            },
            .literal => |l| {
                std.debug.print("literal value={s}\n", .{l.value});
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    return Node{ .name = exp.name, .start = old_pos, .end = pos.* };
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
                return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
            },
            .choice => |s| {
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children) |c| {
                    if (try self.parseInner(&c, data, pos)) |n| {
                        try children.append(n);
                        return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
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
                    return if (l.negative) null else Node{ .name = exp.name, .start = old_pos, .end = old_pos };
                } else {
                    return if (!l.negative) null else Node{ .name = exp.name, .start = old_pos, .end = old_pos };
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
                            return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const gallocator = gpa.allocator();
    var arena = std.heap.ArenaAllocator.init(gallocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var g = Grammar.init(allocator);

    const full_expr = try g.bootstrap();
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
