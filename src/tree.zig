const std = @import("std");

// Utility function, make better later
fn indent(level: u32) void {
    for (0..level) |_| {
        std.debug.print(" ", .{});
    }
}

pub fn NaryTree(comptime T: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,
        nodePool: std.heap.MemoryPool(Node),
        root: ?*Node = null,

        pub const Node = struct {
            value: T,
            children: std.ArrayListUnmanaged(*Node),

            fn init(pool: *std.heap.MemoryPool(Node), value: T) !*Node {
                const n = try pool.create();
                n.* = .{
                    .value = value,
                    .children = std.ArrayListUnmanaged(*Node){},
                };
                return n;
            }

            pub fn addChild(self: *Node, alloc: *std.heap.MemoryPool(Node), value: T) !*Node {
                const child = try Node.init(alloc, value);
                try self.children.append(child);
                return child;
            }

            pub fn deinit(self: *Node, pool: *std.heap.MemoryPool(Node), alloc: std.mem.Allocator) void {
                // Recursively free children
                for (self.children.items) |ch| {
                    ch.deinit(pool, alloc);
                    pool.destroy(ch);
                }
                self.children.deinit(alloc);
            }
        };

        pub fn nodeInit(self: *@This(), value: T) !*Node {
            return Node.init(&self.nodePool, value);
        }

        pub fn nodeAddChild(self: *@This(), n: *Node, value: T) !*Node {
            const new = try self.nodeInit(value);
            try n.children.append(self.allocator, new);
            return new;
        }

        pub fn nodeDeinit(self: *@This(), n: *Node) void {
            n.deinit(&self.nodePool, self.allocator);
            self.nodePool.destroy(n);
        }

        pub fn init(allocator: std.mem.Allocator, root_value: T) !@This() {
            var tree: @This() = .{
                .allocator = allocator,
                .nodePool = std.heap.MemoryPool(Node).init(allocator),
                .root = null,
            };
            tree.root = try tree.nodeInit(root_value);
            return tree;
        }

        pub fn deinit(self: *@This()) void {
            if (self.root) |r| {
                r.deinit(&self.nodePool, self.allocator);
                self.root = null;
            }
            self.nodePool.deinit();
        }

        /// Pre-order DFS; `visitor` is a callable that accepts `*Node` and returns `!void` or `void`.
        pub fn dfs(self: *const @This(), visitor: anytype) !void {
            if (self.root) |r| try dfsNode(r, visitor);
        }
        fn dfsNode(node: *Node, visitor: anytype) !void {
            try visitor(node);
            for (node.children.items) |ch| {
                try dfsNode(ch, visitor);
            }
        }

        /// Breadth-first traversal; `visitor` is a callable like in `dfs`.
        pub fn bfs(self: *const @This(), visitor: anytype) !void {
            if (self.root == null) return;

            var q = std.ArrayList(*Node).init(self.allocator);
            defer q.deinit();

            try q.append(self.root.?);
            var i: usize = 0;
            while (i < q.items.len) : (i += 1) {
                const cur = q.items[i];
                try visitor(cur);
                for (cur.children.items) |ch| try q.append(ch);
            }
        }

        /// Simple stack-based search with a user-provided equality function.
        pub fn find(self: *const @This(), target: T, eql: fn (a: T, b: T) bool) ?*Node {
            if (self.root == null) return null;

            var stack = std.ArrayList(*Node).init(self.allocator);
            defer stack.deinit();

            stack.append(self.root.?) catch return null;
            while (stack.pop()) |node| {
                if (eql(node.value, target)) return node;
                // Push children
                for (node.children.items) |ch| {
                    stack.append(ch) catch return null;
                }
            }
            return null;
        }
    };
}

// -------- Demo / test --------
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const Tree = NaryTree(u32);
    var tree = try Tree.init(alloc, 1);
    defer tree.deinit();

    const r = tree.root.?;
    const a = try r.addChild(2);
    const b = try r.addChild(3);
    _ = try a.addChild(4);
    _ = try a.addChild(5);
    _ = try b.addChild(6);

    const eql = struct {
        fn eq(self: u32, other: u32) bool {
            return self == other;
        }
    }.eq;

    const stdout = std.io.getStdOut().writer();

    try stdout.print("DFS: ", .{});
    try tree.dfs(struct {
        fn visit(n: *Tree.Node) !void {
            std.debug.print("{d} ", .{n.value});
        }
    }.visit);
    try stdout.print("\n", .{});

    try stdout.print("BFS: ", .{});
    try tree.bfs(struct {
        fn visit(n: *Tree.Node) !void {
            std.debug.print("{d} ", .{n.value});
        }
    }.visit);
    try stdout.print("\n", .{});

    if (tree.find(5, eql)) |n| {
        try stdout.print("Found: {d}\n", .{n.value});
    } else {
        try stdout.print("Not found\n", .{});
    }
    var iter = tree.iterator();
    while (iter.next()) |n| {
        std.debug.print("{d} ", .{n.value});
    }
}
