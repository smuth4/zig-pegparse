const std = @import("std");

// Utility function, make better later
fn indent(level: u32) void {
    for (0..level) |_| {
        std.debug.print(" ", .{});
    }
}

// A n-ary tree of nodes, which mananges it's own memory
pub fn NaryTree(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Node = NaryTreeUnmanaged(T).Node;
        const NodePool = std.heap.MemoryPool(Node);

        allocator: std.mem.Allocator,
        nodePool: NodePool,
        unmanaged: NaryTreeUnmanaged(T),

        pub fn init(allocator: std.mem.Allocator, root_value: T) !@This() {
            var pool = NodePool.init(allocator);
            const tree: @This() = .{
                .unmanaged = try NaryTreeUnmanaged(T).init(&pool, root_value),
                .allocator = allocator,
                .nodePool = pool,
            };
            return tree;
        }

        pub fn deinit(self: *@This()) void {
            if (self.unmanaged.root) |r| {
                r.deinit(&self.nodePool, self.allocator);
                self.unmanaged.root = null;
            }
            self.nodePool.deinit();
        }

        pub fn nodeInit(self: *@This(), value: T) !*Node {
            return Node.init(&self.nodePool, value);
        }

        pub fn root(self: *const @This()) ?*Node {
            return self.unmanaged.root;
        }

        pub fn nodeAddChild(self: *@This(), n: *Node, value: T) !*Node {
            const new = try self.nodeInit(value);
            try n.children.append(self.allocator, new);
            return new;
        }

        pub fn nodeAddChildNode(self: *@This(), n: *Node, c: *Node) !*Node {
            try n.children.append(self.allocator, c);
            return c;
        }

        // Note that this does not handle updating the parent to
        // remove the now-invalid pointer, use with care
        pub fn nodeDeinit(self: *@This(), n: *Node) void {
            n.deinit(&self.nodePool, self.allocator);
            self.nodePool.destroy(n);
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
    };
}

// A n-ary tree of nodes, which mananges it's own memory
pub fn NaryTreeUnmanaged(comptime T: type) type {
    return struct {
        const Self = @This();
        root: ?*Node = null,

        const NodePool = std.heap.MemoryPool(Node);
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

        pub fn nodeInit(_: *@This(), pool: *NodePool, value: T) !*Node {
            return Node.init(pool, value);
        }

        pub fn nodeAddChild(self: *@This(), alloc: std.mem.Allocator, n: *Node, value: T) !*Node {
            const new = try self.nodeInit(value);
            try n.children.append(alloc, new);
            return new;
        }

        pub fn nodeSetChild(_: *@This(), alloc: std.mem.Allocator, n: *Node, child: *Node) !void {
            try n.children.append(alloc, child);
        }

        // Note that this does not handle updating the parent to
        // remove the now-invalid pointer
        pub fn nodeDeinit(self: *@This(), n: *Node) void {
            n.deinit(&self.nodePool, self.allocator);
            self.nodePool.destroy(n);
        }

        pub fn init(pool: *NodePool, root_value: T) !@This() {
            var tree: @This() = .{
                .root = null,
            };
            tree.root = try tree.nodeInit(pool, root_value);
            return tree;
        }

        pub fn initRoot(n: *Node) !@This() {
            const tree: @This() = .{
                .root = n,
            };
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

test "Creation and root value" {
    const allocator = std.heap.page_allocator;
    var tree: NaryTree(u32) = try NaryTree(u32).init(allocator, 1);

    const root = tree.root();
    try std.testing.expectEqual(root.?.value, 1);
}

test "Add child node" {
    const allocator = std.heap.page_allocator;
    var tree: NaryTree(u32) = try NaryTree(u32).init(allocator, 1);

    const root = tree.root();
    const child = try tree.nodeAddChild(root.?, 2);

    try std.testing.expectEqual(child.*.value, 2);
    try std.testing.expectEqual(root.?.children.items.len, 1);
}

test "Add multiple child nodes" {
    const allocator = std.heap.page_allocator;
    var tree: NaryTree(u32) = try NaryTree(u32).init(allocator, 1);

    const root = tree.root();
    _ = try tree.nodeAddChild(root.?, 2);
    _ = try tree.nodeAddChild(root.?, 3);
    _ = try tree.nodeAddChild(root.?, 4);

    try std.testing.expectEqual(root.?.children.items.len, 3);
}

test "Deinitialization frees memory" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tree: NaryTree(u32) = try NaryTree(u32).init(allocator, 1);
    const root = tree.root();
    _ = try tree.nodeAddChild(root.?, 2);

    tree.deinit();
}
