const std = @import("std");
const Allocator = std.mem.Allocator;
const zig_pegparse = @import("zig_pegparse");

const regex = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

const NodeList = std.ArrayList(Node);
const Node = struct {
    name: []const u8,
    // Corresponds with the data used to the create the node, which the node isn't actually aware of
    start: usize,
    end: usize,
    children: ?NodeList = null, // null here means a leaf node

    fn print(self: *const Node, data: []const u8, i: u32) void {
        var end = self.end;

        indent(i);
        if (self.name.len != 0) {
            std.debug.print("<Node called \"{s}\" ", .{self.name});
        } else {
            std.debug.print("<Node ", .{});
        }
        if (self.end - self.start > 10) {
            end = self.start + 10;
        }
        if (std.mem.indexOfScalar(u8, data[self.start..end], '\n')) |newlnPos| {
            end = self.start + newlnPos;
        }
        if (end == self.end) {
            std.debug.print("matching \"{s}\">\n", .{data[self.start..self.end]});
        } else {
            std.debug.print("matching \"{s}\"...>\n", .{data[self.start..end]});
        }
        if (self.children) |children| {
            for (children.items) |c| {
                c.print(data, i + 1);
            }
        }
    }
};

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

    const matchData: ?*regex.pcre2_match_data_8 = regex.pcre2_match_data_create_from_pattern_8(regexp, null);
    // regex.PCRE2_ANCHORED prevents us from using JIT compilation, maybe it can be removed somehow?
    const rc: c_int = regex.pcre2_match_8(regexp, subject, subjLen, 0, regex.PCRE2_ANCHORED, matchData.?, null);

    if (rc < 0) {
        return null;
    }

    if (rc == 0) {
        std.debug.print("ovector was not big enough for all the captured substrings\n", .{});
        return null;
    }
    const ovector = regex.pcre2_get_ovector_pointer_8(matchData);
    regex.pcre2_match_data_free_8(matchData);
    // TODO: What is this actually checking? rc is set before ovector

    if (ovector[0] > ovector[1]) {
        std.debug.print("error with ovector\n", .{});
        regex.pcre2_code_free_8(regexp);
        return null;
    }
    return ovector[1] - ovector[0];
}

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

    fn print(self: *const Expression, i: u32) void {
        indent(i);
        switch (self.*.matcher) {
            .regex => |_| {
                std.debug.print("regex name={s}\n", .{self.name});
            },
            .literal => |l| {
                std.debug.print("literal name={s} value={s}\n", .{ self.name, l.value });
            },
            .reference => |r| {
                std.debug.print("reference name={s} target={s}\n", .{ self.name, r.target });
            },
            .sequence => |s| {
                std.debug.print("seq name={s}\n", .{self.name});
                for (s.children.items) |c| {
                    c.print(i + 2);
                }
            },
            .choice => |s| {
                std.debug.print("choice name={s}\n", .{self.name});
                //std.debug.print("choice", .{});
                for (s.children.items) |c| {
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
        //std.debug.print("remaining: {s}\n", .{toParse});
        switch (self.*) {
            .regex => |r| {
                if (find(r.re, toParse)) |result| {
                    const old_pos = pos.*;
                    pos.* += result;
                    return Node{ .name = r.name, .start = old_pos, .end = pos.* };
                } else {
                    return null;
                }
            },
            .literal => |l| {
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    return Node{ .name = self.name, .start = old_pos, .end = pos.* };
                }
            },
            .sequence => |s| {
                var children = NodeList.init(allocator);
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
                var children = NodeList.init(allocator);
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
    root: *Expression,
    // Holds points to expressions for reference lookups
    references: ReferenceTable,
    allocator: Allocator,
    matchCount: usize,

    pub fn init(allocator: Allocator) Grammar {
        return Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = undefined,
            .allocator = allocator,
            .references = ReferenceTable.init(allocator),
            .matchCount = 0,
        };
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
    pub fn parse(self: *Grammar, data: []const u8) !?Node {
        return self.parseWith(data, self.root);
    }

    pub fn parseWith(self: *Grammar, data: []const u8, root: *Expression) !?Node {
        var pos: usize = 0;
        const n = try self.match(root, data, &pos);
        if (pos != data.len) {
            //const start = if (pos > 5) pos - 5 else pos;
            //const end = if (pos < data.len - 6) pos + 5 else data.len - 1;
            //std.debug.print("failed at: {s}\n", .{data[start..end]});
            return null;
        }
        return n;
    }

    // Parse a string and turn it into a new grammar
    pub fn createGrammar(self: *Grammar, data: []const u8) !Grammar {
        var grammar = Grammar.init(self.allocator);
        var visitor = Grammar.ExpressionVisitor{
            .grammar = &grammar,
            .allocator = self.allocator,
            .visitorTable = std.StringHashMap(Grammar.ExpressionVisitor.ExpressionVisitorSignature).init(self.allocator),
            .referenceStack = std.ArrayList([]const u8).init(self.allocator),
        };
        const rootNode = try self.parse(data);
        try visitor.visit(data, &rootNode.?);
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
            std.debug.assert(std.mem.eql(u8, node.name, "rules"));
            var rules = ExpressionList.init(self.allocator);
            const rulesNode = node.children.?.items[1];
            for (rulesNode.children.?.items) |child| {
                //std.debug.print("rule child: {s}\n", .{child.name});
                if (try self.visit_generic(data, &child)) |result| {
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
            if (node.name.len != 0 and node.name[0] == '_') {
                return null;
            }
            if (self.visitorTable.get(node.name)) |func| {
                return func(self, data, node);
            } else {
                if (node.children) |children| {
                    // Return the first non-null result by default
                    for (children.items) |child| {
                        if (node.name.len != 0 and node.name[0] == '_') {
                            return null;
                        }
                        if (try self.visit_generic(data, &child)) |result| {
                            return result;
                        }
                    }
                    return null;
                }
            }
            return null;
        }

        fn visit_sequence(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            if (try self.visit_generic(data, &node.children.?.items[0])) |result| {
                try exprs.append(result);
            }
            for (node.children.?.items[1].children.?.items) |child| {
                if (try self.visit_generic(data, &child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createSequence("", exprs.items);
            return opt_expr;
        }

        fn visit_or_term(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            if (node.children.?.items[2].children.?.items.len == 1) {
                for (node.children.?.items[2].children.?.items) |child| {
                    if (try self.visit_generic(data, &child)) |result| {
                        return result;
                    }
                }
            } else {
                var seq = try self.grammar.createSequence("", &[_]*Expression{});
                for (node.children.?.items[2].children.?.items) |child| {
                    if (try self.visit_generic(data, &child)) |result| {
                        try seq.matcher.sequence.children.append(result);
                    }
                }
                return seq;
            }
            return null;
        }

        fn visit_ored(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            node.print(data, 0);
            // term+
            if (node.children.?.items[0].children.?.items.len == 1) {
                for (node.children.?.items[0].children.?.items) |child| {
                    if (try self.visit_generic(data, &child)) |result| {
                        try exprs.append(result);
                    }
                }
            } else {
                var seq = try self.grammar.createSequence("", &[_]*Expression{});
                for (node.children.?.items[0].children.?.items) |child| {
                    if (try self.visit_generic(data, &child)) |result| {
                        try seq.matcher.sequence.children.append(result);
                    }
                }
                try exprs.append(seq);
            }
            // or_term+
            for (node.children.?.items[1].children.?.items) |child| {
                if (try self.visit_generic(data, &child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createChoice("", exprs.items);
            return opt_expr;
        }

        fn visit_rule(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            if (node.children) |children| {
                const labelExpr = try self.visit_generic(data, &children.items[0]);
                const label = getLiteralValue(labelExpr.?);
                // We could descend and confirm the middle node is '=', but why bother
                const expression = try self.visit_generic(data, &children.items[2]);
                return self.grammar.initExpression(label, expression.?.*.matcher);
            }
            return null;
        }

        // TODO handle escape sequences
        fn visit_double_quoted_literal(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            return try self.grammar.createLiteral("", data[(node.start + 1)..(node.end - 1)]);
        }

        // TODO handle escape sequences
        fn visit_single_quoted_literal(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            return try self.grammar.createLiteral("", data[(node.start + 1)..(node.end - 1)]);
        }

        fn visit_reference(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            const ref_text = try self.visit_generic(data, &node.children.?.items[0]);
            if (data[node.start] == '!') {
                return try self.grammar.createNot("", try self.grammar.createReference("", getLiteralValue(ref_text.?)));
            } else {
                return try self.grammar.createReference("", getLiteralValue(ref_text.?));
            }
        }

        fn visit_quantified(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            const child = try self.visit_generic(data, &node.children.?.items[0]); // Must be a literal
            const q = &node.children.?.items[1].children.?.items[0]; // Quantifier string
            const q_char = data[q.start];
            if (q_char == '*') {
                return try self.grammar.createZeroOrMore("", child.?);
            } else if (q_char == '?') {
                return try self.grammar.createZeroOrOne("", child.?);
            } else if (q_char == '+') {
                return try self.grammar.createOneOrMore("", child.?);
            } else {
                var min: usize = 0;
                var max: usize = std.math.maxInt(usize);
                if (std.mem.indexOfScalarPos(u8, data, q.start, ',')) |comma_pos| {
                    if (comma_pos != q.start + 1) {
                        min = try std.fmt.parseUnsigned(usize, data[q.start + 1 .. comma_pos], 10);
                    }
                    if (comma_pos + 1 != q.end - 1) {
                        max = try std.fmt.parseUnsigned(usize, data[comma_pos + 1 .. q.end - 1], 10);
                    }
                } else {
                    min = try std.fmt.parseUnsigned(usize, data[q.start + 1 .. q.end - 1], 10);
                    max = min;
                }
                return self.grammar.createQuantity("", min, max, child.?);
            }
        }

        fn visit_label_regex(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the label as the name
            if (data[node.start] == '!') {
                return try self.grammar.createNot("", try self.grammar.createLiteral("", data[node.start + 1 .. node.end]));
            } else {
                return try self.grammar.createLiteral("", data[node.start..node.end]);
            }
        }

        fn visit_regex(self: *ExpressionVisitor, data: []const u8, node: *const Node) !?*Expression {
            var options: u32 = 0;
            const optionsNode = node.children.?.items[2];
            const quoted_re = try self.visit_generic(data, &node.children.?.items[1]);
            for (data[optionsNode.start..optionsNode.end]) |c| {
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
                getLiteralValue(quoted_re.?),
                options,
            );
            return re;
        }
    };

    /// Create an expression, storing it in the reference table if it
    /// has a non-empty name, or in the cache otherwise. If neither of
    /// those are possible, it simply gets allocated and returned.
    fn initExpression(self: *Grammar, name: []const u8, matcher: Expression.Matcher) !*Expression {
        const expr = try self.allocator.create(Expression);
        expr.*.name = name;
        expr.*.matcher = matcher;
        if (name.len == 0) {
            switch (matcher) {
                else => return expr,
            }
        }
        const result = try self.references.getOrPut(name);
        if (result.found_existing) {
            //std.debug.print("dupe: {s}\n", .{name});
            return GrammarError.DuplicateLabel;
        } else {
            //std.debug.print("insert reference: {s}\n", .{name});
            result.value_ptr.* = expr;
        }
        return expr;
    }

    // Reduce boilerplate when manually constructing a grammar
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
        var childList = ExpressionList.init(self.allocator);
        for (children) |c| {
            try childList.append(c);
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
        var childList = ExpressionList.init(self.allocator);
        for (children) |c| {
            try childList.append(c);
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
        // Parsimonious bootstraps itself, which is fun, but manually creating the grammar allows us to not have use references
        const ws = try self.createRegex("ws", "\\s+");
        const comment = try self.createRegex("comment", "#[^\r\n]*");
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
        // const rule_data =
        //     \\# Ignored things (represented by _) are typically hung off the end of the
        //     \\# leafmost kinds of nodes. Literals like "/" count as leaves.
        //     \\
        //     \\rules = _ rule*
        //     \\rule = label equals expression
        //     \\equals = "=" _
        //     \\literal = quoted_literal _
        //     \\
        //     \\# So you can't spell a regex like `~"..." ilm`:
        //     \\quoted_literal = ~"\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\"" / ~"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
        //     \\
        //     \\expression = ored / sequence / term
        //     \\or_term = "/" _ term+
        //     \\ored = term+ or_term+
        //     \\sequence = term term+
        //     \\not_term = "!" term _
        //     \\lookahead_term = "&" term _
        //     \\term = not_term / lookahead_term / quantified / atom
        //     \\quantified = atom quantifier
        //     \\atom = reference / literal / regex / parenthesized
        //     \\regex = "~" quoted_literal ~"[ilmsuxa]*"i _
        //     \\parenthesized = "(" _ expression ")" _
        //     \\quantifier = ~"[*+?]|\{\d*,\d+\}|\{\d+,\d*\}|\{\d+\}" _
        //     \\reference = label !equals
        //     \\
        //     \\# A subsequent equal sign is the only thing that distinguishes a label
        //     \\# (which begins a new rule) from a reference (which is just a pointer to a
        //     \\# rule defined somewhere else):
        //     \\label = ~"[a-zA-Z_][a-zA-Z_0-9]*(?![\"'])" _
        //     \\
        //     \\
        //     \\_ = meaninglessness*
        //     \\meaninglessness = ~"\s+" / comment
        //     \\comment = ~"#[^\r\n]*"
        // ;
        // //const rule_data =
        // //    \\reference = label !equals
        // //;
        // const n = try self.parse(rule_data);
        // //n.?.print(rule_data, 0);
        // var visitor = Grammar.BootStrapVisitor{
        //     .grammar = self,
        //     .allocator = self.allocator,
        //     .visitorTable = std.StringHashMap(Grammar.BootStrapVisitor.BootStrapVisitorSignature).init(self.allocator),
        //     .referenceStack = std.ArrayList([]const u8).init(self.allocator),
        // };

        // // Bootstrap against the full rules
        // const root_expr = try visitor.visit(rule_data, &n.?);
        // self.root = root_expr;
        return self.root;
    }

    // Return a tree of Nodes after parsing. Optionals are used to indicate if no match was found.
    pub fn match(self: *Grammar, exp: *const Expression, data: []const u8, pos: *usize) !?Node {
        self.matchCount += 1;
        const toParse = data[pos.*..];
        //if (pos.* != data.len) {
        //    std.debug.print("remaining: {s}\n", .{data[pos.*..@min(pos.* + 10, data.len)]});
        //}
        switch (exp.*.matcher) {
            .regex => |r| {
                // std.debug.print("parse regex name={s} value={s}\n", .{ exp.name, r.value });
                if (find(r.re, toParse)) |result| {
                    //std.debug.print("parse regex match: {s}\n", .{toParse[0..result]});
                    const old_pos = pos.*;
                    pos.* += result;
                    return Node{ .name = exp.name, .start = old_pos, .end = pos.* };
                } else {
                    return null;
                }
            },
            .literal => |l| {
                if (std.mem.startsWith(u8, toParse, l.value)) {
                    //std.debug.print("match literal value={s}\n", .{l.value});
                    const old_pos = pos.*;
                    pos.* += l.value.len;
                    return Node{ .name = exp.name, .start = old_pos, .end = pos.* };
                }
            },
            .reference => |r| {
                //std.debug.print("parse reference target={s}\n", .{r.target});
                if (self.references.get(r.target)) |ref| {
                    return self.match(ref, data, pos);
                }
            },
            .sequence => |s| {
                // TODO: deinit on failure?
                //std.debug.print("parse sequence name={s}\n", .{exp.name});
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children.items) |c| {
                    if (try self.match(c, data, pos)) |n| {
                        try children.append(n);
                    } else {
                        pos.* = old_pos;
                        children.deinit();
                        return null;
                    }
                }
                return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
            },
            .choice => |s| {
                // TODO: deinit on failure?
                //std.debug.print("parse choice name={s}\n", .{exp.name});
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children.items) |c| {
                    if (try self.match(c, data, pos)) |n| {
                        //std.debug.print("parse choice name={s} matched node {s}\n", .{ exp.name, n.name });
                        try children.append(n);
                        return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
                    } else {
                        //std.debug.print("parse choice name={s} failed child {s}\n", .{ exp.name, c.name });
                        pos.* = old_pos;
                    }
                }
                //std.debug.print("parse choice name={s} failed\n", .{exp.name});
                children.deinit();
                return null;
            },
            .lookahead => |l| {
                //std.debug.print("parse lookahead name={s}\n", .{exp.name});
                const old_pos = pos.*;
                const parsedNode = try self.match(l.child, data, pos);
                const new_pos = pos.*;
                pos.* = old_pos; // Always roll back the position
                if (parsedNode) |_| {
                    if (l.negative) {
                        return null;
                    } else {
                        return Node{ .name = exp.name, .start = old_pos, .end = new_pos };
                    }
                } else {
                    return if (!l.negative) null else Node{ .name = exp.name, .start = old_pos, .end = new_pos };
                }
            },
            .quantity => |q| {
                //std.debug.print("parse quantity name={s}\n", .{exp.name});
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                //std.debug.print("quant start: {s}\n", .{exp.name});
                while (children.items.len < q.max and pos.* < data.len) {
                    const parsedNode = try self.match(q.child, data, pos) orelse break;
                    try children.append(parsedNode);
                    //std.debug.print("quant good child: {s}\n", .{parsedNode.name});
                    if (children.items.len >= q.min and parsedNode.start == parsedNode.end) {
                        break;
                    }
                }
                if (children.items.len >= q.min) {
                    //std.debug.print("quant success size: {s} {d}\n", .{ exp.name, children.items.len });
                    return Node{ .name = exp.name, .start = old_pos, .end = pos.*, .children = children };
                } else {
                    pos.* = old_pos;
                    children.deinit();
                    return null;
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

    _ = try g.bootstrap();
    //g.print();
    const json_grammar =
        \\JSON = S? ( Object / Array / String / True / False / Null / Number ) S?
        \\Object = "{"
        \\     String ":" JSON ( "," String ":" JSON )*
        \\ "}"
        \\ObjectPair = String ":" JSON
        \\Array = "[" JSON ( "," JSON )* "]"
        \\String = S? ~'"[^"\\]*(?:\\.[^"\\]*)*"' S?
        \\Escape = ~"\\" ( ~"[bfnrt]" / UnicodeEscape )
        \\UnicodeEscape = "u" ~"[0-9A-Fa-f]{4}"
        \\True = "true"
        \\False = "false"
        \\Null = "null"
        \\Number = Minus? IntegralPart FractionalPart? ExponentPart?
        \\Minus = "-"
        \\IntegralPart = "0" / ~"[1-9]" ~"[0-9]*"
        \\FractionalPart = "." ~"[0-9]+"
        \\ExponentPart = ~"[eE][+-]?" ~"[0-9]+"
        \\S = ~"\s+"
    ;
    //const _ = try g.parse(json_grammar);
    //n.?.print(json_grammar, 0);
    const args = try std.process.argsAlloc(allocator); // Get arguments as a slice
    defer std.process.argsFree(allocator, args); // Free allocated memory
    const cwd = std.fs.cwd();
    const fileContents = try cwd.readFileAlloc(allocator, args[1], std.math.maxInt(usize));
    var g2 = try g.createGrammar(json_grammar);
    //g2.print();
    var pos: usize = 0;
    _ = try g2.match(g2.root, fileContents, &pos);
    std.debug.print("pos: {d} matches: {d}\n", .{ pos, g2.matchCount });

    //n2.?.print("{\"foo\": \"bar\" } ", 0);
}

/////////////
// Testing
/////////////

// A very minimal output format, primarily for testing
fn nodeToString(self: *const Node, output: *std.ArrayList(u8)) !void {
    if (self.name.len > 0 and self.name[0] == '_') {
        return;
    }
    try output.appendSlice(self.name);
    if (self.children) |children| {
        try output.append('[');
        for (children.items) |c| {
            try nodeToString(&c, output);
        }
        try output.append(']');
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
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
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
        .{ .e = try grammar.createRegex("", "test"), .i = "test", .o = "" },
        .{ .e = try grammar.createRegex("", "\\s+"), .i = "     ", .o = "" },
        .{ .e = try grammar.createRegex("", "\\s*"), .i = "", .o = "" },
        .{ .e = try grammar.createOneOrMore("", a), .i = "a", .o = "[a]" },
        .{ .e = try grammar.createOneOrMore("", a), .i = "aaa", .o = "[aaa]" },
        .{ .e = try grammar.createZeroOrMore("", a), .i = "", .o = "[]" },
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
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, try grammar.createLookahead("", b), b }), .i = "ab", .o = "[ab]" },
    };
    var nodeStr = std.ArrayList(u8).init(allocator);
    defer nodeStr.deinit();
    for (cases) |case| {
        const node = try grammar.parseWith(case.i, case.e);
        try nodeToString(&node.?, &nodeStr);
        try std.testing.expectEqualStrings(case.o, nodeStr.items);

        try nodeStr.resize(0);
    }
}

test "expression parse fails" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
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
        .{ .e = try grammar.createRegex("", "test"), .i = "tes" },
        .{ .e = try grammar.createRegex("", "\\s+"), .i = "test" },
        .{ .e = try grammar.createZeroOrOne("", a), .i = "aaa" },
        .{ .e = try grammar.createOneOrMore("", b), .i = "" },
        .{ .e = try grammar.createOneOrMore("", c), .i = "!!!" },
        .{ .e = try grammar.createChoice("", &[_]*Expression{ a, b, c }), .i = "d" },
        .{ .e = try grammar.createSequence("", &[_]*Expression{ a, b }), .i = "ad" },
    };
    for (cases) |case| {
        const node = try grammar.parseWith(case.i, case.e);
        try std.testing.expect(node == null);
    }
}

test "grammar parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var grammar = Grammar.init(allocator);
    _ = try grammar.bootstrap();
    var exprStr = std.ArrayList(u8).init(allocator);
    defer exprStr.deinit();
    const cases = &[_]struct {
        i: []const u8, // input
        o: []const u8, // output
    }{
        .{ .i = "a = a", .o = "rf" },
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
    for (cases) |case| {
        const node = try grammar.parse(case.i);
        node.?.print(case.i, 0);
        try std.testing.expectEqual(node.?.end, case.i.len);
        const new_grammar = try grammar.createGrammar(case.i);
        try expressionToString(new_grammar.root, &exprStr);
        try std.testing.expectEqualStrings(case.o, exprStr.items);

        try exprStr.resize(0);
    }
}
