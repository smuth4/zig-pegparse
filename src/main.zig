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

    pub fn init(allocator: Allocator) Grammar {
        return Grammar{
            // This needs to be filled for parse() to actually do
            // anything, we'll set it with a manually constructed
            // expression for a first bootstrap
            .root = undefined,
            .allocator = allocator,
            .references = ReferenceTable.init(allocator),
        };
    }

    fn print(self: *Grammar) void {
        self.printInner(self.root, 0);
    }

    fn printInner(self: *Grammar, e: *const Expression, i: u32) void {
        indent(i);
        switch (e.*.matcher) {
            .regex => |_| {
                std.debug.print("regex name={s}\n", .{e.name});
            },
            .literal => |l| {
                std.debug.print("literal name={s} value={s}\n", .{ e.name, l.value });
            },
            .reference => |r| {
                std.debug.print("reference name={s} target=\"{s}\"\n", .{ e.name, r.target });
                if (self.references.get(r.target)) |ref| {
                    self.printInner(ref, i + 2);
                } else {
                    indent(i + 2);
                    std.debug.print("undefined!\n", .{});
                }
            },
            .sequence => |s| {
                std.debug.print("seq name={s}\n", .{e.name});
                for (s.children.items) |c| {
                    self.printInner(c, i + 2);
                }
            },
            .choice => |s| {
                std.debug.print("choice name={s}\n", .{e.name});
                //std.debug.print("choice", .{});
                for (s.children.items) |c| {
                    self.printInner(c, i + 2);
                }
            },
            .quantity => |q| {
                std.debug.print("quantity name={s} min={d} max={d}\n", .{ e.name, q.min, q.max });
                self.printInner(q.child, i + 2);
            },
            .lookahead => |l| {
                if (l.negative) {
                    std.debug.print("not name={s}\n", .{e.name});
                } else {
                    std.debug.print("lookahead name={s}\n", .{e.name});
                }
                self.printInner(l.child, i + 2);
            },
        }
    }

    // Parse pre-loaded data according to the grammar
    pub fn parse(self: *Grammar, data: []const u8) !?Node {
        var pos: usize = 0;
        const n = try self.parseInner(self.root, data, &pos);
        if (pos != data.len) {
            std.debug.print("did not reach end\n", .{});
            return null;
        }
        return n;
    }

    // Needs to have access to the grammaer's reference table, but
    // ideally a reference for any visitor implementations
    const BootStrapVisitor = struct {
        const BootStrapVisitorSignature = *const fn (self: *BootStrapVisitor, data: []const u8, node: *const Node) anyerror!?*Expression;
        visitorTable: std.StringHashMap(BootStrapVisitorSignature),
        allocator: Allocator,
        grammar: *Grammar,

        fn visit(self: *BootStrapVisitor, data: []const u8, node: *const Node) !*Expression {
            // Clear this grammar before re-loading any references
            self.grammar.references.clearRetainingCapacity();
            try self.visitorTable.put("regex", &BootStrapVisitor.visit_regex);
            try self.visitorTable.put("rule", &BootStrapVisitor.visit_rule);
            try self.visitorTable.put("label_regex", &BootStrapVisitor.visit_label_regex);
            try self.visitorTable.put("quoted_regex", &BootStrapVisitor.visit_quoted_regex);
            try self.visitorTable.put("sequence", &BootStrapVisitor.visit_sequence);
            try self.visitorTable.put("ored", &BootStrapVisitor.visit_ored);
            try self.visitorTable.put("reference", &BootStrapVisitor.visit_reference);
            std.debug.assert(std.mem.eql(u8, node.name, "rules"));
            var rules = ExpressionList.init(self.allocator);
            //node.print(data, 0);
            const rulesNode = node.children.?.items[1];
            for (rulesNode.children.?.items) |child| {
                std.debug.print("rule child: {s}\n", .{child.name});
                if (try self.visit_generic(data, &child)) |result| {
                    std.debug.print("full rule\n", .{});
                    try rules.append(result);
                }
            }
            return rules.items[0];
        }

        // Kinda gross, might get optimized well but never expose
        fn getLiteralValue(expr: *const Expression) []const u8 {
            switch (expr.matcher) {
                .literal => |l| return l.value,
                .regex => unreachable,
                .sequence => unreachable,
                .choice => unreachable,
                .quantity => unreachable,
                .lookahead => |l| return getLiteralValue(l.child),
                .reference => unreachable,
            }
        }

        fn visit_generic(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
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
                        if (try self.visit_generic(data, &child)) |result| {
                            return result;
                        }
                    }
                    return null;
                }
            }
            return null;
        }

        fn visit_sequence(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            for (node.children.?.items) |child| {
                if (try self.visit_generic(data, &child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createSequence("", exprs.items);
            return opt_expr;
        }

        fn visit_ored(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            var exprs = ExpressionList.init(self.allocator);
            for (node.children.?.items) |child| {
                if (try self.visit_generic(data, &child)) |result| {
                    try exprs.append(result);
                }
            }
            const opt_expr: ?*Expression = try self.grammar.createChoice("", exprs.items);
            return opt_expr;
        }

        fn visit_rule(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            //const rule = self.visit_generic(data, node.children[2], expr);
            node.print(data, 0);
            if (node.children) |children| {
                const labelExpr = try self.visit_generic(data, &children.items[0]);
                const label = getLiteralValue(labelExpr.?);
                // We could descend and confirm the middle node is '=', but why bother
                const expression = try self.visit_generic(data, &children.items[2]);
                return self.grammar.initExpression(label, expression.?.*.matcher);
                //const constructed_expr: ?*Expression = Expression{ .name = label, .matcher = expression.?.matcher };
                //return constructed_expr;
            }
            return null;
        }

        fn visit_quoted_regex(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            return try self.grammar.createLiteral("", data[(node.start + 1)..(node.end - 1)]);
        }

        fn visit_reference(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the data as the name
            const ref_text = try self.visit_generic(data, &node.children.?.items[0]);
            return try self.grammar.createReference("", getLiteralValue(ref_text.?));
        }

        fn visit_label_regex(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            // Send back an empty-value literal with the label as the name
            if (data[node.start] == '!') {
                return try self.grammar.createNot("", try self.grammar.createLiteral("", data[node.start + 1 .. node.end]));
            }
            return try self.grammar.createLiteral("", data[node.start..node.end]);
        }

        fn visit_regex(self: *BootStrapVisitor, data: []const u8, node: *const Node) !?*Expression {
            const quoted_re = try self.visit_generic(data, &node.children.?.items[1]);
            const re = try self.grammar.createRegex(
                "",
                getLiteralValue(quoted_re.?),
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
            std.debug.print("dupe: {s}\n", .{name});
            return GrammarError.DuplicateLabel;
        } else {
            std.debug.print("insert reference: {s}\n", .{name});
            result.value_ptr.* = expr;
        }
        return expr;
    }

    fn addExpression(self: *Grammar, name: []const u8, expr: *Expression) !void {
        // Expressions with empty names can only be referenced directly, but we should either cache them if possible, or just create them with the allocator if not
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
    fn createQuantity(self: *Grammar, name: []const u8, min: usize, max: usize, child: *const Expression) !*Expression {
        return self.initExpression(name, .{ .quantity = Quantity{ .min = min, .max = max, .child = child } });
    }

    fn createZeroOrMore(self: *Grammar, name: []const u8, child: *const Expression) !*Expression {
        return self.createQuantity(name, 0, std.math.maxInt(usize), child);
    }

    fn createOneOrMore(self: *Grammar, name: []const u8, child: *const Expression) !*Expression {
        return self.createQuantity(name, 1, std.math.maxInt(usize), child);
    }

    fn createChoice(self: *Grammar, name: []const u8, children: []const *Expression) !*Expression {
        var childList = ExpressionList.init(self.allocator);
        for (children) |c| {
            try childList.append(c);
        }
        return self.initExpression(name, .{ .choice = Choice{ .children = childList } });
    }

    fn createLookahead(self: *Grammar, name: []const u8, child: *const Expression) !*Expression {
        return self.initExpression(name, .{ .lookahead = Lookahead{ .negative = false, .child = child } });
    }

    fn createNot(self: *Grammar, name: []const u8, child: *const Expression) !*Expression {
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
        const re = compile(value) orelse return GrammarError.InvalidRegex;
        return self.initExpression(name, .{ .regex = Regex{ .value = value, .re = re } });
    }

    fn createLiteral(self: *Grammar, name: []const u8, value: []const u8) !*Expression {
        return self.initExpression(name, .{ .literal = Literal{ .value = value } });
    }

    fn createReference(self: *Grammar, name: []const u8, target: []const u8) !*Expression {
        return self.initExpression(name, .{ .reference = Reference{ .target = target } });
    }

    pub fn bootstrap(self: *Grammar) !*Expression {
        // Build a basic grammar for bootstrapping
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
        const quoted_literal = try self.createSequence(
            "quoted_literal",
            &[_]*Expression{
                try self.createRegex("quoted_regex", "\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\""),
                ignore,
            },
        );
        const regex_exp = try self.createSequence("regex", &[_]*Expression{ try self.createLiteral("", "~"), quoted_literal });
        const reference = try self.createSequence("reference", &[_]*Expression{ label, try self.createNot("", equals) });
        const atom = try self.createChoice("atom", &[_]*Expression{
            reference,
            quoted_literal,
            regex_exp,
        });
        const quantifier = try self.createSequence("quantifier", &[_]*Expression{ try self.createRegex("", "[*+?]"), ignore });
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
            ignore,
            try self.createOneOrMore("or_term_plus", or_term),
        });
        const expression = try self.createChoice("expression", &[_]*Expression{ ored, sequence, term });
        const rule = try self.createSequence("rule", &[_]*Expression{
            label,
            equals,
            expression,
        });
        const rules = try self.createSequence("rules", &[_]*Expression{ ignore, try self.createZeroOrMore("", rule) });

        // Assign the rules to ourselves
        self.root = rules;
        //self.root.print(0);
        const rule_data =
            \\# Ignored things (represented by _) are typically hung off the end of the
            \\# leafmost kinds of nodes. Literals like "/" count as leaves.
            \\
            \\rules = _ rule*
            \\rule = label equals expression
            \\equals = "=" _
            \\literal = spaceless_literal _
            \\
            \\# So you can't spell a regex like `~"..." ilm`:
            \\spaceless_literal = ~"\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\""is /
            \\                    ~"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"is
            \\
            \\expression = ored / sequence / term
            \\or_term = "/" _ term+
            \\ored = term+ or_term+
            \\sequence = term term+
            \\not_term = "!" term _
            \\lookahead_term = "&" term _
            \\term = not_term / lookahead_term / quantified / atom
            \\quantified = atom quantifier
            \\atom = reference / literal / regex / parenthesized
            \\regex = "~" spaceless_literal ~"[ilmsuxa]*"i _
            \\parenthesized = "(" _ expression ")" _
            \\quantifier = ~"[*+?]|\{\d*,\d+\}|\{\d+,\d*\}|\{\d+\}" _
            \\reference = label !equals
            \\
            \\# A subsequent equal sign is the only thing that distinguishes a label
            \\# (which begins a new rule) from a reference (which is just a pointer to a
            \\# rule defined somewhere else):
            \\label = ~"[a-zA-Z_][a-zA-Z_0-9]*(?![\"'])" _
            \\ 
            \\
            \\_ = meaninglessness*
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
        self.root = root_expr;
        return root_expr;
    }

    // Return a tree of Nodes after parsing. Optionals are used to indicate if no match was found.
    fn parseInner(self: *Grammar, exp: *const Expression, data: []const u8, pos: *usize) !?Node {
        const toParse = data[pos.*..];
        std.debug.print("remaining: {s}\n", .{toParse});
        switch (exp.*.matcher) {
            .regex => |r| {
                std.debug.print("regex name={s}\n", .{exp.name});
                if (find(r.re, toParse)) |result| {
                    // Regexes are one of the only types that can
                    // match 0-length nodes, ignore those
                    if (result == 0) {
                        return null;
                    }
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
            .reference => |r| {
                std.debug.print("reference target={s}\n", .{r.target});
                if (self.references.get(r.target)) |ref| {
                    return self.parseInner(ref, data, pos);
                }
            },
            .sequence => |s| {
                var children = std.ArrayList(Node).init(self.allocator);
                const old_pos = pos.*;
                for (s.children.items) |c| {
                    if (try self.parseInner(c, data, pos)) |n| {
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
                for (s.children.items) |c| {
                    if (try self.parseInner(c, data, pos)) |n| {
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

    _ = try g.bootstrap();
    std.debug.print("bootstrapped\n", .{});
    var keyItr = g.references.keyIterator();
    while (keyItr.next()) |key| {
        std.debug.print("registered ref: {s}\n", .{key.*});
    }
    g.print();
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
