const std = @import("std");
const Allocator = std.mem.Allocator;
const zig_pegparse = @import("zig_pegparse");

const tracy = @import("tracy.zig");
const trace = tracy.trace;

const ntree = @import("tree.zig");

const regex = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

const Span = struct {
    start: usize,
    end: usize,
    expr: *const Expression,
};

const SpanTree = ntree.NaryTree(Span);

const Node = SpanTree.Node;

// Global match data to be re-used for regexes
var match_data: ?*regex.pcre2_match_data_8 = null;

fn get_match_data() ?*regex.pcre2_match_data_8 {
    if (match_data) |m| {
        return m;
    } else {
        match_data = regex.pcre2_match_data_create_8(1, null);
        return match_data;
    }
}

/// Compiles a regex pattern string and returns a pattern code you can use
/// to match subjects. Returns `null` if something is wrong with the pattern
fn compile(needle: []const u8, options: u32) ?*regex.pcre2_code_8 {
    const pattern: regex.PCRE2_SPTR8 = needle.ptr;
    const patternLen: regex.PCRE2_SIZE = needle.len;
    var errornumber: c_int = undefined;
    var erroroffset: regex.PCRE2_SIZE = undefined;

    const regexp: ?*regex.pcre2_code_8 = regex.pcre2_compile_8(pattern, patternLen, options, &errornumber, &erroroffset, null);
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
    const subject: regex.PCRE2_SPTR8 = haystack.ptr;
    const subjLen: regex.PCRE2_SIZE = haystack.len;

    // regex.PCRE2_ANCHORED prevents us from using JIT compilation, maybe it can be removed somehow?
    const rc: c_int = regex.pcre2_match_8(regexp, subject, subjLen, 0, regex.PCRE2_ANCHORED, get_match_data(), null);

    if (rc < 0) {
        return null;
    }

    if (rc == 0) {
        std.debug.print("ovector was not big enough for all the captured substrings\n", .{});
        return null;
    }
    const ovector = regex.pcre2_get_ovector_pointer_8(get_match_data());

    if (ovector[0] > ovector[1]) {
        std.debug.print("error with ovector\n", .{});
        return null;
    }
    return ovector[1] - ovector[0];
}

// Data structures for supported expressions
const ExpressionList = std.ArrayList(*Expression);
const Literal = struct {
    value: []const u8,
};

const Reference = struct {
    target: []const u8,
};

const Regex = struct {
    value: []const u8,
    re: *regex.pcre2_code_8,
};

const Sequence = struct {
    children: ExpressionList,
};

const Choice = struct {
    children: ExpressionList,
};

// Handles ?, *, +, and {min,max}
const Quantity = struct {
    min: usize,
    max: usize,
    child: *Expression,
};

// Handles both ! and &
const Lookahead = struct {
    negative: bool = false,
    child: *Expression,
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
        reference: Reference,
    };

    name: []const u8, // Name can't be changed once created
    matcher: Matcher,
};

const ReferenceTable = std.StringHashMap(*Expression);
const ReferenceList = std.ArrayList(*Expression);

const Grammar = struct {
    // The main allocator, will also be used for expressionArena and the
    // generated node tree by default.
    allocator: Allocator,
    // Where parsing defaults to starting from
    root: *Expression,
    // Holds points to expressions for reference lookups
    references: ReferenceTable,
    // Expressions can be recursive and possibly orphans, but also are
    // relatively small, hence a dedicated arena for them.
    expressionArena: std.heap.ArenaAllocator,
    ignorePrefix: u8 = '_',
    // Some debugging stats

    pub fn init(allocator: Allocator) Grammar {
        return Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = undefined,
            .allocator = allocator,
            .expressionArena = std.heap.ArenaAllocator.init(allocator),
            .references = ReferenceTable.init(allocator),
        };
    }

    pub fn createFactory(allocator: Allocator) Grammar {
        var g = Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = undefined,
            .allocator = allocator,
            .expressionArena = std.heap.ArenaAllocator.init(allocator),
            .references = ReferenceTable.init(allocator),
        };
        g.bootstrap();
        return g;
    }

    pub fn deinit(self: *Grammar) void {
        self.references.clearAndFree();
        self.expressionArena.deinit();
    }

    fn print(self: *const Grammar) void {
        var referenceStack = std.ArrayList([]const u8).init(self.allocator);
        self.printInner(&referenceStack, self.root, 0);
    }

    fn printInner(self: *const Grammar, rs: *std.ArrayList([]const u8), e: *const Expression, i: u32) void {
        indent(i);
        switch (e.*.matcher) {
            .regex => |r| {
                std.debug.print("regex name={s} value={s}\n", .{ e.name, r.value });
            },
            .literal => |l| {
                std.debug.print("literal name={s} value={s}\n", .{ e.name, l.value });
            },
            .reference => |r| {
                std.debug.print("reference name={s} target=\"{s}\"\n", .{ e.name, r.target });
                if (self.references.get(r.target)) |ref| {
                    for (rs.items) |rsc| {
                        if (std.mem.eql(u8, rsc, r.target)) {
                            indent(i + 2);
                            std.debug.print("terminating cyclic reference!\n", .{});
                            return;
                        }
                    }
                    rs.append(r.target) catch {};
                    self.printInner(rs, ref, i + 2);
                    _ = rs.pop().?;
                } else {
                    indent(i + 2);
                    std.debug.print("undefined!\n", .{});
                }
            },
            .sequence => |s| {
                std.debug.print("seq name={s}\n", .{e.name});
                for (s.children.items) |c| {
                    self.printInner(rs, c, i + 2);
                }
            },
            .choice => |s| {
                std.debug.print("choice name={s}\n", .{e.name});
                //std.debug.print("choice", .{});
                for (s.children.items) |c| {
                    self.printInner(rs, c, i + 2);
                }
            },
            .quantity => |q| {
                std.debug.print("quantity name={s} min={d} max={d}\n", .{ e.name, q.min, q.max });
                self.printInner(rs, q.child, i + 2);
            },
            .lookahead => |l| {
                if (l.negative) {
                    std.debug.print("not name={s}\n", .{e.name});
                } else {
                    std.debug.print("lookahead name={s}\n", .{e.name});
                }
                self.printInner(rs, l.child, i + 2);
            },
        }
    }

    // Parse pre-loaded data according to the grammar
    pub fn parse(self: *Grammar, data: []const u8) !?SpanTree {
        return self.parseWith(data, self.root);
    }

    pub fn parseWith(self: *Grammar, data: []const u8, root: *Expression) !?SpanTree {
        var pos: usize = 0;
        var tree = try SpanTree.init(self.allocator, .{ .expr = root, .start = 0, .end = 0 });
        try self.match(root, data, &pos, &tree, tree.root.?);
        if (pos != data.len) {
            //const start = if (pos > 5) pos - 5 else pos;
            //const end = if (pos < data.len - 6) pos + 5 else data.len - 1;
            //std.debug.print("failed at: {s}\n", .{data[start..end]});
            return null;
        }
        return tree;
    }

    // Parse a string and turn it into a new grammar
    pub fn createGrammar(self: *Grammar, data: []const u8) !Grammar {
        var grammar = Grammar.init(self.allocator);
        var visitor = Grammar.ExpressionVisitor{
            .grammar = &grammar,
            .allocator = grammar.allocator,
            .visitorTable = std.StringHashMap(Grammar.ExpressionVisitor.ExpressionVisitorSignature).init(grammar.allocator),
            .referenceStack = std.ArrayList([]const u8).init(grammar.allocator),
        };
        defer visitor.visitorTable.deinit();
        var tree = try self.parse(data);
        defer {
            if (tree != null) { // Can't use `if () |_|` because it doesn't detect deinit() as a
                tree.?.deinit();
            }
        }
        try visitor.visit(data, tree.?.root.?);
        return grammar;
    }

    /// Converts a node tree into a full grammar
    const ExpressionVisitor = struct {
        const ExpressionVisitorSignature = *const fn (self: *ExpressionVisitor, data: []const u8, node: *const Node) anyerror!?*Expression;
        visitorTable: std.StringHashMap(ExpressionVisitorSignature),
        allocator: Allocator,
        grammar: *Grammar,
        referenceStack: std.ArrayList([]const u8),

        fn visit(self: *ExpressionVisitor, data: []const u8, node: *const Node) !void {
            // Clear this grammar before re-loading any references
            self.grammar.references.clearRetainingCapacity();
            try self.visitorTable.put("regex", &ExpressionVisitor.visit_regex);
            try self.visitorTable.put("rule", &ExpressionVisitor.visit_rule);
            try self.visitorTable.put("label_regex", &ExpressionVisitor.visit_label_regex);
            try self.visitorTable.put("single_quoted_literal", &ExpressionVisitor.visit_double_quoted_literal);
            try self.visitorTable.put("double_quoted_literal", &ExpressionVisitor.visit_single_quoted_literal);
            try self.visitorTable.put("sequence", &ExpressionVisitor.visit_sequence);
            try self.visitorTable.put("ored", &ExpressionVisitor.visit_ored);
            try self.visitorTable.put("or_term", &ExpressionVisitor.visit_or_term);
            try self.visitorTable.put("reference", &ExpressionVisitor.visit_reference);
            try self.visitorTable.put("quantified", &ExpressionVisitor.visit_quantified);
            //std.debug.assert(std.mem.eql(u8, node.value.name, "rules"));
            var rules = ExpressionList.init(self.allocator);
            defer rules.deinit();
            const rulesNode = node.children.items[0].children.items[1];
            for (rulesNode.children.items) |child| {
                //std.debug.print("rule child: {s}\n", .{child.name});
                if (try self.visit_generic(data, child)) |result| {
                    //self.grammar.printInner(&self.referenceStack, result, 0);
                    try rules.append(result);
                }
            }
            self.grammar.root = rules.items[0];
        }

        // Kinda gross, might get optimized well but never expose
        fn getLiteralValue(expr: *const Expression) []const u8 {
            switch (expr.matcher) {
                .literal => |l| return l.value,
                .lookahead => |l| return getLiteralValue(l.child),
                // All these are just a lazy form of debugging the type
                .regex => unreachable,
                .sequence => unreachable,
                .choice => unreachable,
                .quantity => unreachable,
                .reference => unreachable,
            }
        }

        fn visit_generic(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Skip over anything starting with '_'
            if (node.value.expr.*.name.len != 0 and node.value.expr.*.name[0] == '_') {
                return null;
            }
            if (self.visitorTable.get(node.value.expr.*.name)) |func| {
                return func(self, data, node);
            } else {
                // Return the first non-null result by default
                for (node.children.items) |child| {
                    if (node.value.expr.*.name.len != 0 and node.value.expr.*.name[0] == '_') {
                        return null;
                    }
                    if (try self.visit_generic(data, child)) |result| {
                        return result;
                    }
                }
                return null;
            }
            return null;
        }

        fn visit_sequence(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            defer exprs.deinit();
            if (try self.visit_generic(data, node.children.items[0])) |result| {
                try exprs.append(result);
            }
            for (node.children.items[1].children.items) |child| {
                if (try self.visit_generic(data, child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createSequence("", exprs.items);
            return opt_expr;
        }

        fn visit_or_term(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            if (node.children.items[2].children.items.len == 1) {
                for (node.children.items[2].children.items) |child| {
                    if (try self.visit_generic(data, child)) |result| {
                        return result;
                    }
                }
            } else {
                var seq = try self.grammar.createSequence("", &[_]*Expression{});
                for (node.children.items[2].children.items) |child| {
                    if (try self.visit_generic(data, child)) |result| {
                        try seq.matcher.sequence.children.append(result);
                    }
                }
                return seq;
            }
            return null;
        }

        fn visit_ored(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            defer exprs.deinit();
            // term+
            if (node.children.items[0].children.items.len == 1) {
                if (try self.visit_generic(data, node.children.items[0].children.items[0])) |result| {
                    try exprs.append(result);
                }
            } else {
                var seq = try self.grammar.createSequence("", &[_]*Expression{});
                for (node.children.items[0].children.items) |child| {
                    if (try self.visit_generic(data, child)) |result| {
                        try seq.matcher.sequence.children.append(result);
                    }
                }
                try exprs.append(seq);
            }
            // or_term+
            for (node.children.items[1].children.items) |child| {
                if (try self.visit_generic(data, child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createChoice("", exprs.items);
            return opt_expr;
        }

        fn visit_rule(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            const labelExpr = try self.visit_generic(data, node.children.items[0]);
            const label = getLiteralValue(labelExpr.?);
            // We could descend and confirm the middle node is '=', but why bother
            const expression = try self.visit_generic(data, node.children.items[2]);
            return self.grammar.initExpression(label, expression.?.*.matcher);
        }

        // TODO handle escape sequences
        fn visit_double_quoted_literal(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            var unescaped_literal = std.ArrayList(u8).init(self.grammar.*.expressionArena.allocator());
            _ = try std.zig.string_literal.parseWrite(unescaped_literal.writer(), data[(node.value.start + 1)..(node.value.end - 1)]);
            return try self.grammar.createLiteral("", try unescaped_literal.toOwnedSlice());
        }

        // TODO handle escape sequences
        fn visit_single_quoted_literal(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            return try self.grammar.createLiteral("", data[(node.value.start + 1)..(node.value.end - 1)]);
        }

        fn visit_reference(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            const ref_text = try self.visit_generic(data, node.children.items[0]);
            if (data[node.value.start] == '!') {
                return try self.grammar.createNot("", try self.grammar.createReference("", getLiteralValue(ref_text.?)));
            } else {
                return try self.grammar.createReference("", getLiteralValue(ref_text.?));
            }
        }

        fn visit_quantified(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            const child = try self.visit_generic(data, node.children.items[0]); // Must be a literal
            const q = node.children.items[1].children.items[0]; // Quantifier string
            const q_char = data[q.value.start];
            if (q_char == '*') {
                return try self.grammar.createZeroOrMore("", child.?);
            } else if (q_char == '?') {
                return try self.grammar.createZeroOrOne("", child.?);
            } else if (q_char == '+') {
                return try self.grammar.createOneOrMore("", child.?);
            } else {
                var min: usize = 0;
                var max: usize = std.math.maxInt(usize);
                if (std.mem.indexOfScalarPos(u8, data, q.value.start, ',')) |comma_pos| {
                    if (comma_pos != q.value.start + 1) {
                        min = try std.fmt.parseUnsigned(usize, data[q.value.start + 1 .. comma_pos], 10);
                    }
                    if (comma_pos + 1 != q.value.end - 1) {
                        max = try std.fmt.parseUnsigned(usize, data[comma_pos + 1 .. q.value.end - 1], 10);
                    }
                } else {
                    min = try std.fmt.parseUnsigned(usize, data[q.value.start + 1 .. q.value.end - 1], 10);
                    max = min;
                }
                return self.grammar.createQuantity("", min, max, child.?);
            }
        }

        fn visit_label_regex(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the label as the name
            if (data[node.value.start] == '!') {
                return try self.grammar.createNot("", try self.grammar.createLiteral("", data[node.value.start + 1 .. node.value.end]));
            } else {
                return try self.grammar.createLiteral("", data[node.value.start..node.value.end]);
            }
        }

        fn visit_regex(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var options: u32 = 0;
            const optionsNode = node.children.items[2];
            const re_string = node.children.items[1].children.items[0];
            const quoted_re = data[re_string.value.start + 1 .. re_string.value.end - 1];
            for (data[optionsNode.value.start..optionsNode.value.end]) |c| {
                options = options | switch (c) {
                    'i' => regex.PCRE2_CASELESS,
                    'm' => regex.PCRE2_MULTILINE,
                    's' => regex.PCRE2_DOTALL,
                    'u' => regex.PCRE2_UTF,
                    'a' => regex.PCRE2_NEVER_UTF,
                    else => 0,
                };
            }
            const re = try self.grammar.createRegexOptions(
                "",
                quoted_re,
                options,
            );
            return re;
        }
    };

    /// Create an expression, storing it in the reference table if it
    /// has a non-empty name, or in the cache otherwise. If neither of
    /// those are possible, it simply gets allocated and returned.
    fn initExpression(self: *Grammar, name: []const u8, matcher: Expression.Matcher) !*Expression {
        const expr = try self.expressionArena.allocator().create(Expression);
        expr.*.name = name;
        expr.*.matcher = matcher;
        if (name.len != 0) {
            const result = try self.references.getOrPut(name);
            if (result.found_existing) {
                //std.debug.print("dupe: {s}\n", .{name});
                return GrammarError.DuplicateLabel;
            } else {
                //std.debug.print("insert reference: {s}\n", .{name});
                result.value_ptr.* = expr;
            }
        }
        return expr;
    }

    // Utitility functions for creating various expressions
    fn createQuantity(self: *Grammar, name: []const u8, min: usize, max: usize, child: *Expression) !*Expression {
        return self.initExpression(name, .{ .quantity = Quantity{ .min = min, .max = max, .child = child } });
    }

    fn createZeroOrOne(self: *Grammar, name: []const u8, child: *Expression) !*Expression {
        return self.createQuantity(name, 0, 1, child);
    }

    fn createZeroOrMore(self: *Grammar, name: []const u8, child: *Expression) !*Expression {
        return self.createQuantity(name, 0, std.math.maxInt(usize), child);
    }

    fn createOneOrMore(self: *Grammar, name: []const u8, child: *Expression) !*Expression {
        return self.createQuantity(name, 1, std.math.maxInt(usize), child);
    }

    fn createChoice(self: *Grammar, name: []const u8, children: []const *Expression) !*Expression {
        var childList = ExpressionList.init(self.expressionArena.allocator());
        try childList.ensureTotalCapacity(children.len);
        for (children) |c| {
            childList.appendAssumeCapacity(c);
        }
        return self.initExpression(name, .{ .choice = Choice{ .children = childList } });
    }

    fn createLookahead(self: *Grammar, name: []const u8, child: *Expression) !*Expression {
        return self.initExpression(name, .{ .lookahead = Lookahead{ .negative = false, .child = child } });
    }

    fn createNot(self: *Grammar, name: []const u8, child: *Expression) !*Expression {
        return self.initExpression(name, .{ .lookahead = Lookahead{ .negative = true, .child = child } });
    }

    fn createSequence(self: *Grammar, name: []const u8, children: []const *Expression) !*Expression {
        var childList = ExpressionList.init(self.expressionArena.allocator());
        try childList.ensureTotalCapacity(children.len);
        for (children) |c| {
            childList.appendAssumeCapacity(c);
        }
        return self.initExpression(name, .{ .sequence = Sequence{ .children = childList } });
    }

    fn createRegex(self: *Grammar, name: []const u8, value: []const u8) !*Expression {
        return self.createRegexOptions(name, value, 0);
    }

    fn createRegexOptions(self: *Grammar, name: []const u8, value: []const u8, options: u32) !*Expression {
        const re = compile(value, options) orelse return GrammarError.InvalidRegex;
        return self.initExpression(name, .{ .regex = Regex{ .value = value, .re = re } });
    }

    fn createLiteral(self: *Grammar, name: []const u8, value: []const u8) !*Expression {
        return self.initExpression(name, .{ .literal = Literal{ .value = value } });
    }

    fn createReference(self: *Grammar, name: []const u8, target: []const u8) !*Expression {
        return self.initExpression(name, .{ .reference = Reference{ .target = target } });
    }

    pub fn bootstrap(self: *Grammar) !*Expression {
        // Allow PCRE2 to reserve some memory before anything else
        _ = get_match_data();

        // Parsimonious bootstraps itself, which is fun, but manually
        // creating the grammar allows us to not have use references
        const ws = try self.createRegex("_ws", "\\s+");
        const comment = try self.createRegex("_comment", "#[^\r\n]*");
        const ignore = try self.createZeroOrMore("_ignore", try self.createChoice(
            "",
            &[_]*Expression{ ws, comment },
        ));
        const equals = try self.createSequence("equals", &[_]*Expression{ try self.createLiteral("", "="), ignore });
        const label = try self.createSequence(
            "label",
            &[_]*Expression{
                try self.createRegex("label_regex", "!?[a-zA-Z_][a-zA-Z_0-9]*"),
                ignore,
            },
        );
        const double_quoted_literal = try self.createRegex("double_quoted_literal",
            \\"[^"\\]*(?:\\.[^"\\]*)*"
        );
        const single_quoted_literal = try self.createRegex("single_quoted_literal",
            \\'[^'\\]*(?:\\.[^'\\]*)*'
        );
        const quoted_literal = try self.createSequence(
            "quoted_literal",
            &[_]*Expression{
                try self.createChoice("", &[_]*Expression{ double_quoted_literal, single_quoted_literal }),
                ignore,
            },
        );
        const regex_exp = try self.createSequence("regex", &[_]*Expression{
            try self.createLiteral("", "~"),
            try self.createChoice("quoted_regexp", &[_]*Expression{ double_quoted_literal, single_quoted_literal }),
            try self.createRegex("", "[imsua]*"),
            ignore,
            //ignore,
        });
        const parenthesized = try self.createSequence("parenthesized", &[_]*Expression{
            try self.createLiteral("", "("),
            ignore,
            try self.createReference("", "expression"), // TODO: Use a direct pointer
            ignore,
            try self.createLiteral("", ")"),
            ignore,
        });
        const reference = try self.createSequence("reference", &[_]*Expression{ label, try self.createNot("", equals) });
        const atom = try self.createChoice("atom", &[_]*Expression{
            reference,
            quoted_literal,
            regex_exp,
            parenthesized,
        });
        const quantifier = try self.createSequence("quantifier", &[_]*Expression{
            try self.createRegex("",
                \\[*+?]|\{\d*,\d+\}|\{\d+,\d*\}|\{\d+\}
            ),
            ignore,
        });
        const quantified = try self.createSequence("quantified", &[_]*Expression{ atom, quantifier });
        const term = try self.createChoice("term", &[_]*Expression{ quantified, atom });
        const term_plus = try self.createOneOrMore("term_plus", term); // deviation from parsimonious
        const sequence = try self.createSequence("sequence", &[_]*Expression{ term, term_plus });
        const or_term = try self.createSequence("or_term", &[_]*Expression{
            try self.createLiteral("", "/"),
            ignore,
            term_plus,
        });
        const ored = try self.createSequence("ored", &[_]*Expression{
            term_plus,
            try self.createOneOrMore("or_term_plus", or_term),
        });
        const expression = try self.createChoice("expression", &[_]*Expression{ ored, sequence, term });
        const rule = try self.createSequence("rule", &[_]*Expression{
            label,
            equals,
            expression,
        });
        const rules = try self.createSequence("rules", &[_]*Expression{ ignore, try self.createOneOrMore("", rule) });

        // Assign the rules to ourselves
        self.root = rules;
        return self.root;
    }

    // Start matching `data` with `exp` starting from `pos`, adding children under `node` in `tree`
    pub fn match(self: *Grammar, exp: *const Expression, data: []const u8, pos: *usize, tree: *SpanTree, node: *SpanTree.Node) !void {
        const toParse = data[pos.*..];
        switch (exp.*.matcher) {
            .regex => |r| {
                //std.debug.print("parse regex name={s} value={s}\n", .{ exp.name, r.value });
                if (find(r.re, toParse)) |result| {
                    //std.debug.print("parse regex match: {s}\n", .{toParse[0..result]});
                    const old_pos = pos.*;
                    pos.* += result;
                    _ = try tree.nodeAddChild(node, .{ .expr = exp, .start = old_pos, .end = pos.* });
                }
            },
            .literal => |l| {
                //std.debug.print("literal value={s}\n", .{l.value});
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    //std.debug.print("match literal value={s}\n", .{l.value});
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    _ = try tree.nodeAddChild(node, .{ .expr = exp, .start = old_pos, .end = pos.* });
                }
            },
            .reference => |r| {
                //std.debug.print("parse reference target={s}\n", .{r.target});
                if (self.references.get(r.target)) |ref| {
                    return self.match(ref, data, pos, tree, node);
                }
            },
            .sequence => |s| {
                //std.debug.print("parse sequence name={s}\n", .{exp.name});
                var child = try tree.nodeAddChild(node, .{ .expr = exp, .start = pos.*, .end = pos.* });

                for (s.children.items, 1..) |c, i| {
                    try self.match(c, data, pos, tree, child);
                    if (child.children.items.len != i) {
                        // Failure, deinit, reset pos and exit
                        pos.* = child.value.start;
                        tree.nodeDeinit(child);
                        _ = node.children.pop();
                        break;
                    }
                }
                child.value.end = pos.*;
            },
            .choice => |s| {
                //std.debug.print("parse choice name={s}\n", .{exp.name});
                var child = try tree.nodeAddChild(node, .{ .expr = exp, .start = pos.*, .end = pos.* });

                for (s.children.items) |c| {
                    try self.match(c, data, pos, tree, child);
                    if (child.children.items.len > 0) {
                        // Success
                        child.value.end = pos.*;
                        break;
                    }
                } else {
                    // No matches, reset
                    pos.* = child.value.start;
                    tree.nodeDeinit(child);
                    _ = node.children.pop();
                }
            },
            .lookahead => |l| {
                //std.debug.print("parse lookahead name={s}\n", .{exp.name});
                var child = try tree.nodeAddChild(node, .{ .expr = exp, .start = pos.*, .end = pos.* });

                try self.match(l.child, data, pos, tree, child);
                // Always roll back
                pos.* = child.value.start;

                const found_match = (child.children.items.len == 0) == l.negative;
                //std.debug.print("parse lookahead state l:{d} n:{s} m:{s}\n", .{ child.children.items.len, if (l.negative) "neg" else "pos", if (found_match) "yes" else "no" });
                if (found_match) {
                    if (child.children.items.len > 0) {
                        child.value.end = child.children.items[0].value.end;
                    }
                } else {
                    tree.nodeDeinit(child);
                    _ = node.children.pop();
                }
            },
            .quantity => |q| {
                // std.debug.print("parse quantity name={s}\n", .{exp.name});
                var child = try tree.nodeAddChild(node, .{ .expr = exp, .start = pos.*, .end = pos.* });
                var i: usize = 0; //Expected count of children
                while (child.children.items.len < q.max and pos.* < data.len) {
                    i += 1;
                    try self.match(q.child, data, pos, tree, child);
                    // std.debug.print("parse quantity state l:{d} i:{d}, q.min:{d} q.max:{d}\n", .{ child.children.items.len, i, q.min, q.max });
                    // Couldn't find a next match, or exceeded the minimum
                    if (child.children.items.len != i or child.children.items.len >= q.max) {
                        break;
                    }
                    // Avoid long loops on empty repeating matches
                    const lastValue = child.children.items[i - 1].value;
                    if (lastValue.start == lastValue.end) {
                        break;
                    }
                    // std.debug.print("parse quantity continue\n", .{});
                }
                // std.debug.print("parse quantity end state l:{d} i:{d}, q.min:{d} q.max:{d}\n", .{ child.children.items.len, i, q.min, q.max });
                if (child.children.items.len < q.min) {
                    tree.nodeDeinit(child);
                    _ = node.children.pop();
                } else {
                    if (child.children.items.len > 0) {
                        child.value.end = child.children.items[child.children.items.len - 1].value.end;
                    }
                }
            },
        }
    }

    fn optimize(self: *Grammar) void {
        self.optimizeInner(self.root);
    }

    fn optimizeInner(self: *Grammar, exp: *Expression) void {
        switch (exp.*.matcher) {
            .choice => |c| {
                for (c.children.items) |*child| {
                    switch (child.*.matcher) {
                        // Cut out any references with direct pointers
                        .reference => |r| {
                            if (self.references.get(r.target)) |*ref| {
                                child.* = ref.*;
                            }
                        },
                        else => {
                            self.optimizeInner(child.*);
                        },
                    }
                }
            },
            .quantity => |c| {
                var child = c.child;
                switch (child.*.matcher) {
                    // Cut out any references with direct pointers
                    .reference => |r| {
                        if (self.references.get(r.target)) |*ref| {
                            child = ref.*;
                        }
                    },
                    else => {
                        self.optimizeInner(child);
                    },
                }
            },
            .sequence => |c| {
                for (c.children.items) |*child| {
                    switch (child.*.matcher) {
                        // Cut out any references with direct pointers
                        .reference => |r| {
                            if (self.references.get(r.target)) |*ref| {
                                child.* = ref.*;
                            }
                        },
                        else => {
                            self.optimizeInner(child.*);
                        },
                    }
                }
            },
            else => {},
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var g = Grammar.init(allocator);
    defer g.deinit();

    _ = try g.bootstrap();
    //g.print();
    const json_grammar =
        \\JSON = _S? ( String / Array / Object / True / False / Null / Number ) _S?
        \\Object = "{"
        \\     String ":" JSON ( "," String ":" JSON )*
        \\ "}"
        \\Array = "[" JSON ( "," JSON )* "]"
        \\String = _S? ~'"[^"\\]*(?:\\.[^"\\]*)*"' _S?
        \\Escape = ~"\\" ( ~"[bfnrt]" / UnicodeEscape )
        \\UnicodeEscape = "u" ~"[0-9A-Fa-f]{4}"
        \\True = "true"
        \\False = "false"
        \\Null = "null"
        \\Number = ~"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
        \\Minus = "-"
        \\IntegralPart = "0" / ~"[1-9][0-9]*"
        \\FractionalPart = ~"\.[0-9]+"
        \\ExponentPart = ~"[eE][+-]?" ~"[0-9]+"
        \\_S = ~"\s+"
    ;

    const args = try std.process.argsAlloc(allocator); // Get arguments as a slice
    defer std.process.argsFree(allocator, args); // Free allocated memory

    const cwd = std.fs.cwd();
    const fileContents = try cwd.readFileAlloc(allocator, args[1], std.math.maxInt(usize));
    defer allocator.free(fileContents);

    var g2 = try g.createGrammar(json_grammar);
    defer g2.deinit();

    g2.optimize();
    var pos: usize = 0;

    const start_time = try std.time.Instant.now();
    var t = try SpanTree.init(allocator, .{ .expr = g2.root, .start = 0, .end = 0 });
    try g2.match(g2.root, fileContents, &pos, &t, t.root.?);
    const end_time = try std.time.Instant.now();
    const elapsed_nanos = end_time.since(start_time);
    std.debug.print("Elapsed time: {d} ms\n", .{elapsed_nanos / 1_000_000});

    // std.debug.print("pos: {d} matches: {d}\n", .{ pos, g2.matchCount });
    t.deinit();
    //std.debug.print("tree count: {d} node count: {d}\n", .{ t.root.?.count(), g2.nodeCount });
    //std.debug.print("struct size: {d}\n", .{@sizeOf(SpanTree.Node)});
}

/////////////
// Testing
/////////////

// A very minimal output format, primarily for testing
fn nodeToString(self: *const Node, output: *std.ArrayList(u8)) !void {
    if (self.value.name.len > 0 and self.value.name[0] == '_') {
        return;
    }
    try output.appendSlice(self.value.name);

    if (self.children.items.len > 0) {
        try output.append('[');
        for (self.children.items) |c| {
            try nodeToString(c, output);
        }
        try output.append(']');
    }
}

// Utility function for expressionToRhs
fn usizeToStr(num: usize, output: *std.ArrayList(u8)) !void {
    var i = num;
    if (i == 0) {
        try output.append('0');
    } else {
        while (i > 0) {
            const digit = @as(u8, @intCast(i % 10));
            try output.append('0' + digit);
            i /= 10;
        }
    }
}

fn expressionToRhs(self: *const Expression, output: *std.ArrayList(u8)) !void {
    switch (self.*.matcher) {
        .regex => |r| {
            try output.appendSlice("~\"");
            try output.appendSlice(r.value);
            try output.append('"');
        },
        .literal => |l| {
            try output.append('"');
            try output.appendSlice(l.value);
            try output.append('"');
        },
        .reference => |r| {
            try output.appendSlice(r.target);
        },
        .sequence => |s| {
            try output.appendSlice("( ");
            for (s.children.items, 0..) |c, i| {
                try expressionToRhs(c, output);
                if (i != s.children.items.len - 1) {
                    try output.append(' ');
                }
            }
            try output.appendSlice(" ) ");
        },
        .choice => |s| {
            try output.appendSlice("( ");
            for (s.children.items, 0..) |c, i| {
                try expressionToRhs(c, output);
                if (i != s.children.items.len - 1) {
                    try output.appendSlice(" / ");
                }
            }
            try output.appendSlice(" ) ");
        },
        .quantity => |q| {
            try expressionToRhs(q.child, output);
            if (q.min == 0 and q.max == 1) {
                try output.append('?');
            } else if (q.min == 0 and q.max == std.math.maxInt(usize)) {
                try output.append('*');
            } else if (q.min == 1 and q.max == std.math.maxInt(usize)) {
                try output.append('+');
            } else {
                try output.append('{');
                if (q.min == q.max) {
                    try usizeToStr(q.min, output);
                } else {
                    if (q.min != 0) {
                        try usizeToStr(q.min, output);
                    }
                    try output.append(',');
                    if (q.max != std.math.maxInt(usize)) {
                        try usizeToStr(q.max, output);
                    }
                }
                try output.append('}');
            }
        },
        .lookahead => |l| {
            if (l.negative) {
                try output.append('!');
            } else {
                try output.append('&');
            }
            try expressionToRhs(l.child, output);
        },
    }
}

// A very minimal output format, primarily for testing
fn expressionToString(self: *const Expression, output: *std.ArrayList(u8)) !void {
    switch (self.*.matcher) {
        .regex => {
            try output.appendSlice("rx");
        },
        .literal => {
            try output.appendSlice("l");
        },
        .reference => {
            try output.appendSlice("rf");
        },
        .sequence => |s| {
            try output.appendSlice("s[");
            for (s.children.items) |c| {
                try expressionToString(c, output);
            }
            try output.append(']');
        },
        .choice => |s| {
            try output.appendSlice("c[");
            for (s.children.items) |c| {
                try expressionToString(c, output);
            }
            try output.append(']');
        },
        .quantity => |q| {
            try output.appendSlice("q[");
            try expressionToString(q.child, output);
            try output.append(']');
        },
        .lookahead => |l| {
            if (l.negative) {
                try output.appendSlice("n[");
            } else {
                try output.appendSlice("la[");
            }
            try expressionToString(l.child, output);
            try output.append(']');
        },
    }
}

test "expressions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var grammar = Grammar.init(allocator);
    // Create a few labels ahead of time to avoid dupes
    const a = try grammar.createLiteral("a", "a");
    const b = try grammar.createLiteral("b", "b");
    const c = try grammar.createLiteral("c", "c");
    const cases = &[_]struct {
        e: *Expression,
        i: []const u8, // input
        o: []const u8, // output from nodes' toString
    }{
        .{ .e = try grammar.createLiteral("", "="), .i = "=", .o = "" },
        .{ .e = try grammar.createLiteral("", "test"), .i = "test", .o = "" },
        .{ .e = try grammar.createLiteral("", "\\n"), .i = "\n", .o = "" },
        .{ .e = try grammar.createRegex("", "test"), .i = "test", .o = "" },
        .{ .e = try grammar.createRegex("", "\\s+"), .i = "     ", .o = "" },
        .{ .e = try grammar.createRegex("", "\\s*"), .i = "", .o = "" },
        .{ .e = try grammar.createOneOrMore("", a), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createOneOrMore("", a), .i = "aaa", .o = "[aaa]" },
        .{ .e = try grammar.createZeroOrMore("", a), .i = "", .o = "" }, // Matched but no children, maybe a separate test?
        .{ .e = try grammar.createZeroOrMore("", a), .i = "aaa", .o = "[aaa]" },
        .{ .e = try grammar.createZeroOrOne("", a), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{a}), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, b }), .i = "ab", .o = "[ab]" },
        .{ .e = try grammar.createChoice("", &[_]*Expression{ a, b, c }), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createChoice("", &[_]*Expression{ a, b, c }), .i = "b", .o = "[b]" },
        .{ .e = try grammar.createChoice("", &[_]*Expression{ a, b, c }), .i = "c", .o = "[c]" },
        .{ .e = try grammar.createReference("", "a"), .i = "a", .o = "a" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, try grammar.createNot("", b) }), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, try grammar.createNot("", b), c }), .i = "ac", .o = "[ac]" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, try grammar.createLookahead("", b), b }), .i = "ab", .o = "[a[b]b]" },
    };

    var nodeStr = std.ArrayList(u8).init(allocator);
    defer nodeStr.deinit();

    for (cases) |case| {
        const tree = try grammar.parseWith(case.i, case.e);
        try nodeToString(tree.?.root.?.children.items[0], &nodeStr);
        try std.testing.expectEqualStrings(case.o, nodeStr.items);

        try nodeStr.resize(0);
    }
}

test "expression parse fails" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var grammar = Grammar.init(allocator);
    // Create a few labels ahead of time to avoid dupes
    const a = try grammar.createLiteral("a", "a");
    const b = try grammar.createLiteral("b", "b");
    const c = try grammar.createLiteral("c", "c");
    const cases = &[_]struct {
        e: *Expression,
        i: []const u8, // input
    }{
        .{ .e = a, .i = "!" },
        .{ .e = try grammar.createLiteral("", "test"), .i = "test2" },
        .{ .e = try grammar.createLiteral("", "test"), .i = "tes" },
        .{ .e = try grammar.createLiteral("", "\\n"), .i = "\t" },
        .{ .e = try grammar.createRegex("", "test"), .i = "tes" },
        .{ .e = try grammar.createRegex("", "\\s+"), .i = "test" },
        .{ .e = try grammar.createZeroOrOne("", a), .i = "aaa" },
        .{ .e = try grammar.createOneOrMore("", c), .i = "!!!" },
        .{ .e = try grammar.createChoice("", &[_]*Expression{ a, b, c }), .i = "d" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, b }), .i = "ad" },
    };
    for (cases) |case| {
        const tree = try grammar.parseWith(case.i, case.e);
        try std.testing.expect(tree == null);
    }
}

test "grammar parsing" {
    const cases = &[_]struct {
        i: []const u8, // input
        o: []const u8, // output
    }{
        .{ .i = "a = a", .o = "rf" },
        .{ .i = "a  =      a", .o = "rf" },
        .{ .i = "a = a # comment", .o = "rf" },
        .{ .i = "a = \"x\"", .o = "l" },
        .{ .i = "a = ~\"x\"", .o = "rx" },
        .{ .i = "a = ~'x'", .o = "rx" },
        .{ .i = "a = ~\"x\"i", .o = "rx" },
        .{ .i = "a = ~\"x\"is", .o = "rx" },
        .{ .i = "a = ~\"x\"i", .o = "rx" },
        .{ .i = "a = 'a'", .o = "l" },
        .{ .i = "a = !b", .o = "n[rf]" },
        .{ .i = "a = b*", .o = "q[rf]" },
        .{ .i = "a = \"x\"*", .o = "q[l]" },
        .{ .i = "a = b+", .o = "q[rf]" },
        .{ .i = "a = b?", .o = "q[rf]" },
        .{ .i = "a = b{2,3}", .o = "q[rf]" },
        .{ .i = "a = b{2,}", .o = "q[rf]" },
        .{ .i = "a = b{,3}", .o = "q[rf]" },
        .{ .i = "a = b{3}", .o = "q[rf]" },
        .{ .i = "a = a b c", .o = "s[rfrfrf]" },
        .{ .i = "a = a / b / c", .o = "c[rfrfrf]" },
        .{ .i = "a = ( a b c )", .o = "s[rfrfrf]" },

        .{ .i = "a = a ( a / c )", .o = "s[rfc[rfrf]]" },
        .{ .i = "a = a / b c", .o = "c[rfs[rfrf]]" },
        .{ .i = "a = a / b / c / a", .o = "c[rfrfrfrf]" },
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var grammar = Grammar.init(allocator);
    _ = try grammar.bootstrap();

    var exprStr = std.ArrayList(u8).init(allocator);
    defer exprStr.deinit();
    for (cases) |case| {
        const tree = try grammar.parse(case.i);
        try std.testing.expectEqual(case.i.len, tree.?.root.?.children.items[0].value.end);
        const new_grammar = try grammar.createGrammar(case.i);
        try expressionToString(new_grammar.root, &exprStr);
        try std.testing.expectEqualStrings(case.o, exprStr.items);

        try exprStr.resize(0);
    }
}
