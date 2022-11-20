use graph::Graph;
use itertools::Itertools;

const MIN_NODE_NUM: usize = 4;
const MAX_NODE_NUM: usize = 100;
const NODE_NUM_RANGE: std::ops::RangeInclusive<usize> = MIN_NODE_NUM..=MAX_NODE_NUM;
const QUERY_NUM: usize = 100;

pub static mut GRAPH_NUM: Option<usize> = None;
pub static mut NOISE_RATE: Option<usize> = None;

pub fn set_graph_num(graph_num: usize) {
    unsafe {
        debug_assert!(GRAPH_NUM.is_none());

        GRAPH_NUM = Some(graph_num);
    }
}

pub fn get_graph_num() -> usize {
    unsafe { GRAPH_NUM.expect("No value set.") }
}

pub fn set_noise_rate(noise_rate: usize) {
    unsafe {
        debug_assert!(NOISE_RATE.is_none());

        NOISE_RATE = Some(noise_rate);
    }
}

pub fn get_noise_rate() -> usize {
    unsafe { NOISE_RATE.expect("No value set.") }
}

pub fn get_epsilon() -> f64 {
    get_noise_rate() as f64 / 100.0
}

fn main() {
    for node_num in 1..=7 {
        println!("{}", String::from("*").repeat(80));
        show_graphs_info(node_num);
        println!("{}", String::from("*").repeat(80));
        println!();
    }
}

pub fn show_graphs_info(node_num: usize) {
    let (labeled_graphs, unique_graphs) = Graph::create_labeled_graphs(node_num);

    println!("node_num = {}", node_num);
    println!(
        "calc_edge_domain(node_num) = {}",
        calc_edge_domain(node_num)
    );
    println!("unique_graphs.len() = {}", unique_graphs.len());
    println!("labeled_graph.len() = {}", labeled_graphs.len());

    let mut classified_graphs = vec![vec![]; unique_graphs.len()];
    for (graph, label) in labeled_graphs {
        classified_graphs[label].push(graph);
    }

    let calc_min_max_edge_num = |graphs: &Vec<Graph>| {
        graphs
            .iter()
            .map(|graph| {
                (0..node_num)
                    .map(|node_idx| graph.edge_num(node_idx))
                    .max()
                    .unwrap()
            })
            .min()
            .unwrap()
    };

    classified_graphs.sort_by_key(|graphs| calc_min_max_edge_num(graphs));

    let max_min_max_edge_num = classified_graphs
        .iter()
        .take(100)
        .map(|graphs| calc_min_max_edge_num(graphs))
        .max()
        .unwrap();
    println!(
        "max_min_max_edge_num = {} (graph_num = 100)",
        max_min_max_edge_num
    );
}

pub fn calc_edge_domain(node_num: usize) -> usize {
    node_num * (node_num - 1) / 2
}

pub fn calc_edge_idx(node_num: usize, from: usize, to: usize) -> usize {
    debug_assert!(from < to);

    calc_edge_domain(node_num) * from - from * (from + 1) / 2 + to - 1
}

pub fn calc_acc_edge_domain(node_num: usize) -> Vec<usize> {
    (0..node_num)
        .rev()
        .scan(0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect_vec()
}

pub mod union_find {
    //! Union-Find processes the following queries for an edgeless graph in `O(α(n))` amortized time.
    //! * Add an undirected edge.
    //! * Deciding whether given two vertices are in the same connected component
    //!
    //! When a method is called, route compression is performed as appropriate.

    use std::collections::HashMap;

    /// Union-Find processes the following queries for an edgeless graph in `O(α(n))` amortized time.
    /// * Add an undirected edge.
    /// * Deciding whether given two vertices are in the same connected component
    ///
    /// When a method is called, route compression is performed as appropriate.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::union_find::UnionFind;
    ///
    /// let mut uf = UnionFind::new(3);
    /// assert_eq!(uf.same(0, 2), false);
    /// uf.merge(0, 1);
    /// assert_eq!(uf.same(0, 2), false);
    /// uf.merge(1, 2);
    /// assert_eq!(uf.same(0, 2), true);
    /// ```
    pub struct UnionFind {
        /// For each element, one of the following is stored.
        /// * Size of the connected component to which it belongs; It is expressed by a negative number
        /// (if it is representative of a connected component)
        /// * Index of the element that is its own parent (otherwise)
        parent_or_size: Vec<i32>,

        /// Number of connected components.
        group_num: usize,
    }

    impl UnionFind {
        /// Creates an undirected graph with `n` vertices and 0 edges.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(0, 1);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(2, 1);
        /// assert_eq!(uf.same(0, 2), true);
        /// ```
        pub fn new(n: usize) -> Self {
            UnionFind {
                parent_or_size: vec![-1; n],
                group_num: n,
            }
        }

        /// Returns the representative of the connected component that contains the vertex `a`.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// uf.merge(1, 2);
        /// assert_eq!(uf.leader(0), 0);
        /// assert_eq!(uf.leader(1), uf.leader(2));
        /// ```
        pub fn leader(&mut self, a: usize) -> usize {
            // Path from A to just before the representative
            // of the connected component to which A belongs
            // (If the representative is A, then it is empty)
            let mut path = vec![];

            // Variable representing the current vertex (initialized with a)
            let mut curr = a;

            // Loop until the vertex indicated by curr becomes the parent.
            while self.parent_or_size[curr] >= 0 {
                // Add curr to the path.
                path.push(curr);
                // Move to parent vertex.
                curr = self.parent_or_size[curr] as usize;
            }

            // Set the parent of every vertex in the path to representative of the connected component.
            path.iter()
                .for_each(|&x| self.parent_or_size[x] = curr as i32);

            // Return a representative of the connected component.
            curr
        }

        /// Returns whether the vertices `a` and `b` are in the same connected component.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(0, 1);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(2, 1);
        /// assert_eq!(uf.same(0, 2), true);
        /// ```
        pub fn same(&mut self, a: usize, b: usize) -> bool {
            self.leader(a) == self.leader(b)
        }

        /// Adds an edge between vertex `a` and vertex `b`.
        /// Returns true if the connected component to which a belongs and that of b are newly combined.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(0, 1);
        /// assert_eq!(uf.same(0, 2), false);
        /// uf.merge(2, 1);
        /// assert_eq!(uf.same(0, 2), true);
        /// ```
        pub fn merge(&mut self, a: usize, b: usize) -> bool {
            // Representative of the connected component that contains the vertex a
            let mut leader_a = self.leader(a);
            // Representative of the connected component that contains the vertex b
            let mut leader_b = self.leader(b);

            // If a and b belong to the same connected component, return false without processing.
            if leader_a == leader_b {
                return false;
            }

            // If the size of the connected component to which a belongs is
            // smaller than that of b, swap a and b.
            if -self.parent_or_size[leader_a] < -self.parent_or_size[leader_b] {
                std::mem::swap(&mut leader_a, &mut leader_b);
            }

            // Add that of b to the number of elements of the connected component to which a belongs.
            self.parent_or_size[leader_a] += self.parent_or_size[leader_b];

            // Set the parent of the representative of the connected component to which b belongs
            // to the representative of the connected component to which a belongs.
            self.parent_or_size[leader_b] = leader_a as i32;

            // Decrease the number of connected components by one.
            self.group_num -= 1;

            // Return true because the connected component is newly combined.
            true
        }

        /// Returns a list of connected components.
        ///
        /// Each list consists of the indexes of the vertices
        /// of the corresponding connected component.
        /// The lists are arranged in ascending order with respect to
        /// the smallest index contained in the list.
        /// The indexes contained in each list are arranged in ascending order.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(5);
        /// uf.merge(1, 2);
        /// uf.merge(2, 3);
        /// assert_eq!(uf.groups(), vec![vec![0], vec![1, 2, 3], vec![4]]);
        /// ```
        pub fn groups(&mut self) -> Vec<Vec<usize>> {
            let mut leader_to_idx: HashMap<usize, usize> = HashMap::new();
            let mut groups: Vec<Vec<usize>> = vec![];

            for i in 0..self.parent_or_size.len() {
                let leader = self.leader(i);

                if let Some(&idx) = leader_to_idx.get(&leader) {
                    groups[idx].push(i);
                } else {
                    leader_to_idx.insert(leader, groups.len());
                    groups.push(vec![i]);
                }
            }

            groups
        }

        /// Returns the size of the connected component that contains the vertex `a`.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// assert_eq!(uf.size(0), 1);
        /// uf.merge(0, 1);
        /// assert_eq!(uf.size(0), 2);
        /// uf.merge(2, 1);
        /// assert_eq!(uf.size(0), 3);
        /// ```
        pub fn size(&mut self, a: usize) -> usize {
            let leader = self.leader(a);
            -self.parent_or_size[leader] as usize
        }

        /// Adds a new vertex.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(4);
        /// uf.merge(1, 2);
        /// uf.merge(2, 3);
        /// assert_eq!(uf.groups(), vec![vec![0], vec![1, 2, 3]]);
        /// uf.add();
        /// assert_eq!(uf.groups(), vec![vec![0], vec![1, 2, 3], vec![4]]);
        /// ```
        pub fn add(&mut self) {
            self.parent_or_size.push(-1);
            self.group_num += 1;
        }

        /// Returns the number of connected components.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(3);
        /// assert_eq!(uf.group_num(), 3);
        /// uf.merge(0, 1);
        /// assert_eq!(uf.group_num(), 2);
        /// uf.merge(2, 1);
        /// assert_eq!(uf.group_num(), 1);
        /// ```
        pub fn group_num(&self) -> usize {
            self.group_num
        }

        /// Returns the number of elements.
        ///
        /// # Examples
        ///
        /// ```
        /// use atcoder8_library::union_find::UnionFind;
        ///
        /// let mut uf = UnionFind::new(5);
        /// assert_eq!(uf.elem_num(), 5);
        /// ```
        pub fn elem_num(&self) -> usize {
            self.parent_or_size.len()
        }
    }
}

pub mod graph {
    use std::{
        collections::{HashMap, HashSet},
        io::{self, Write},
    };

    use itertools::{sorted, Itertools};
    use rand::seq::SliceRandom;

    use crate::{calc_edge_domain, get_noise_rate, union_find::UnionFind};

    #[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
    pub struct Node {
        edges: u128,
        edge_num: usize,
    }

    impl Node {
        pub fn new(edges: u128) -> Self {
            Self {
                edges,
                edge_num: edges.count_ones() as usize,
            }
        }

        fn edge_num(&self) -> usize {
            self.edge_num
        }

        fn is_connect(&self, other_node: usize) -> bool {
            (self.edges >> other_node) & 1 == 1
        }

        fn edges(&self, node_num: usize) -> Vec<usize> {
            (0..node_num)
                .filter(|&node_idx| self.is_connect(node_idx))
                .collect()
        }

        fn add_edge(&mut self, other_node: usize) -> bool {
            if self.is_connect(other_node) {
                return false;
            }

            self.edges |= 1 << other_node;
            self.edge_num += 1;

            true
        }
    }

    #[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
    pub struct Graph {
        nodes: Vec<Node>,
        total_edge_num: usize,
    }

    impl Graph {
        pub fn new(node_num: usize) -> Self {
            Self {
                nodes: vec![Node::default(); node_num],
                total_edge_num: 0,
            }
        }

        pub fn from_chars(hh: &Vec<char>, node_num: usize) -> Self {
            let mut graph = Graph::new(node_num);

            let mut from = 0;
            let mut to = 1;

            for &h in hh {
                if h == '1' {
                    graph.add_edge(from, to);
                }

                to += 1;
                if to == node_num {
                    from += 1;
                    to = from + 1;
                }
            }

            graph
        }

        pub fn from_edge_num(node_num: usize, edge_num: usize) -> Self {
            let mut graph = Graph::new(node_num);

            let mut from = 0;
            let mut to = 1;

            for _ in 0..edge_num {
                graph.add_edge(from, to);

                to += 1;
                if to == node_num {
                    from += 1;
                    to = from + 1;
                }
            }

            graph
        }

        pub fn from_balanced_edges(node_num: usize, edge_num: usize) -> Self {
            assert!(edge_num <= calc_edge_domain(node_num));

            let mut graph = Graph::new(node_num);

            let mut from = 0;
            let mut dist = 1;

            for _ in 0..edge_num {
                graph.add_edge(from, (from + dist) % node_num);

                from += 1;
                if from == node_num {
                    from = 0;
                    dist += 1;
                }
            }

            graph
        }

        pub fn from_edge_num_with_random<R>(node_num: usize, edge_num: usize, rng: &mut R) -> Self
        where
            R: rand::Rng,
        {
            let mut candidate = (0..(node_num - 1))
                .flat_map(|from| ((from + 1)..node_num).map(move |to| (from, to)))
                .collect_vec();
            candidate.shuffle(rng);

            let mut graph = Graph::new(node_num);

            for &(from, to) in candidate.iter().take(edge_num) {
                graph.add_edge(from, to);
            }

            graph
        }

        pub fn add_edge(&mut self, from: usize, to: usize) -> bool {
            debug_assert_ne!(from, to);

            if self.is_connect(from, to) {
                return false;
            }

            self.nodes[from].add_edge(to);
            self.nodes[to].add_edge(from);
            self.total_edge_num += 1;

            true
        }

        pub fn node_num(&self) -> usize {
            self.nodes.len()
        }

        pub fn edge_domain(&self) -> usize {
            self.node_num() * (self.node_num() - 1) / 2
        }

        pub fn total_edge_num(&self) -> usize {
            self.total_edge_num
        }

        pub fn edge_num(&self, node_idx: usize) -> usize {
            self.nodes[node_idx].edge_num()
        }

        pub fn expected_edge_num_diff(&self) -> f64 {
            let eps = get_noise_rate() as f64 / 100.0;
            let expected_dec = self.total_edge_num as f64 * eps;
            let expected_inc = (self.edge_domain() - self.total_edge_num) as f64 * eps;

            expected_inc - expected_dec
        }

        pub fn expected_edge_num(&self) -> f64 {
            self.total_edge_num as f64 + self.expected_edge_num_diff()
        }

        pub fn is_connect(&self, from: usize, to: usize) -> bool {
            assert_eq!(
                self.nodes[from].is_connect(to),
                self.nodes[to].is_connect(from)
            );

            self.nodes[from].is_connect(to)
        }

        pub fn edges(&self, node_idx: usize) -> Vec<usize> {
            self.nodes[node_idx].edges(self.node_num())
        }

        pub fn show_edges(&self) {
            let node_num = self.node_num();

            let mut from = 0;
            let mut to = 1;

            while from < node_num - 1 {
                print!("{}", if self.is_connect(from, to) { '1' } else { '0' });

                to += 1;
                if to == node_num {
                    from += 1;
                    to = from + 1;
                }
            }

            println!();

            io::stdout().flush().unwrap();
        }

        pub fn permutation(&self, indexes: &Vec<usize>) -> Graph {
            let node_num = self.node_num();

            assert!((0..node_num).zip(sorted(indexes)).all(|(x, y)| x == *y));

            let mut dest_graph = Graph::new(node_num);

            for from in 0..(node_num - 1) {
                for to in (from + 1)..node_num {
                    if self.is_connect(from, to) {
                        dest_graph.add_edge(indexes[from], indexes[to]);
                    }
                }
            }

            dest_graph
        }

        pub fn permutations(&self) -> Vec<Graph> {
            (0..self.node_num())
                .permutations(self.node_num())
                .map(|indexes| self.permutation(&indexes))
                .collect()
        }

        pub fn create_labeled_graphs(node_num: usize) -> (HashMap<Graph, usize>, Vec<Graph>) {
            let edge_domain = calc_edge_domain(node_num);

            let mut dp: Vec<Option<Graph>> = vec![None; 1 << edge_domain];
            let mut labeled_graphs: HashMap<Graph, usize> = HashMap::new();
            dp[0] = Some(Graph::new(node_num));
            labeled_graphs.insert(Graph::new(node_num), 0);
            let mut class_cnt = 1;

            for bit in 0..(1 << edge_domain) {
                let mut edge_idx = 0;

                for from in 0..(node_num - 1) {
                    for to in (from + 1)..node_num {
                        let next_bit = bit | (1 << edge_idx);

                        if let Some(cur_graph) = &dp[bit] {
                            if !cur_graph.is_connect(from, to) {
                                let mut next_graph = cur_graph.clone();
                                next_graph.add_edge(from, to);

                                if !labeled_graphs.contains_key(&next_graph) {
                                    labeled_graphs.extend(
                                        next_graph
                                            .permutations()
                                            .into_iter()
                                            .map(|graph| (graph, class_cnt)),
                                    );
                                    class_cnt += 1;
                                    dp[next_bit] = Some(next_graph);
                                }
                            }
                        }

                        edge_idx += 1;
                    }
                }
            }

            let mut unique_graphs = vec![Graph::new(node_num); class_cnt];
            for graph in dp {
                if let Some(graph) = graph {
                    let class = *labeled_graphs.get(&graph).unwrap();
                    unique_graphs[class] = graph;
                }
            }

            (labeled_graphs, unique_graphs)
        }

        pub fn create_unique_graphs(node_num: usize) -> Vec<Graph> {
            let edge_domain = calc_edge_domain(node_num);

            let mut graph_pool: HashSet<Graph> = HashSet::new();

            let mut dp: Vec<Option<Graph>> = vec![None; 1 << edge_domain];
            dp[0] = Some(Graph::new(node_num));

            for bit in 0..(1 << edge_domain) {
                let mut edge_idx = 0;

                for from in 0..(node_num - 1) {
                    for to in (from + 1)..node_num {
                        let next_bit = bit | (1 << edge_idx);

                        if let Some(cur_graph) = &dp[bit] {
                            if !cur_graph.is_connect(from, to) {
                                let mut next_graph = cur_graph.clone();
                                next_graph.add_edge(from, to);

                                if !graph_pool.contains(&next_graph) {
                                    graph_pool.extend(next_graph.permutations());
                                    dp[next_bit] = Some(next_graph);
                                }
                            }
                        }

                        edge_idx += 1;
                    }
                }
            }

            dp.into_iter().filter_map(|graph| graph).collect()
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct GraphPartInfo {
        node_num: usize,
        sorted_group_sizes: Vec<usize>,
        sorted_degrees: Vec<usize>,
    }

    impl GraphPartInfo {
        pub fn new(graph: &Graph) -> Self {
            let node_num = graph.node_num();

            let mut uf = UnionFind::new(graph.node_num());
            for from in 0..(node_num - 1) {
                for to in (from + 1)..node_num {
                    uf.merge(from, to);
                }
            }

            let mut degrees = (0..node_num)
                .map(|node_idx| graph.edge_num(node_idx))
                .collect_vec();
            degrees.sort_unstable();

            let mut group_sizes = (0..node_num)
                .filter_map(|node_idx| {
                    if uf.leader(node_idx) == node_idx {
                        Some(uf.size(node_idx))
                    } else {
                        None
                    }
                })
                .collect_vec();
            group_sizes.sort_unstable();

            Self {
                node_num: graph.node_num(),
                sorted_group_sizes: group_sizes,
                sorted_degrees: degrees,
            }
        }

        pub fn sorted_group_sizes(&self) -> &Vec<usize> {
            &self.sorted_group_sizes
        }

        pub fn sorted_degrees(&self) -> &Vec<usize> {
            &self.sorted_degrees
        }

        pub fn expected_sorted_degrees(&self) -> Vec<f64> {
            let epsilon = get_noise_rate() as f64 / 100.0;

            self.sorted_degrees
                .iter()
                .map(|&degree| {
                    degree as f64 + epsilon * ((self.node_num - 1) as f64 - 2.0 * degree as f64)
                })
                .collect()
        }

        pub fn energy_by_degrees(&self, received_graph_part_info: &GraphPartInfo) -> f64 {
            self.expected_sorted_degrees()
                .iter()
                .zip(received_graph_part_info.sorted_degrees())
                .map(|(&expected_degree, &received_degree)| {
                    (expected_degree - received_degree as f64).powi(2)
                })
                .sum()
        }
    }
}

pub mod without_noise {
    use std::{
        collections::HashMap,
        io::{self, Write},
    };

    use itertools::Itertools;

    use crate::{
        calc_edge_domain, get_graph_num,
        graph::{Graph, GraphPartInfo},
        QUERY_NUM,
    };

    fn create_distinguishable_graphs(node_num: usize) -> Option<HashMap<GraphPartInfo, Graph>> {
        io::stdout().flush().unwrap();

        let graph_num = get_graph_num();
        let edge_domain = calc_edge_domain(node_num);

        let mut distinguishable_graphs = HashMap::new();

        for edges in 0..(1_u128 << edge_domain) {
            if distinguishable_graphs.len() >= graph_num {
                break;
            }

            let mut graph = Graph::new(node_num);

            let mut from = 0;
            let mut to = 1;

            for edge_idx in 0..edge_domain {
                if (edges >> edge_idx) & 1 == 1 {
                    graph.add_edge(from, to);
                }

                to += 1;
                if to == node_num {
                    from += 1;
                    to = from + 1;
                }
            }

            let graph_part_info = GraphPartInfo::new(&graph);

            if !distinguishable_graphs.contains_key(&graph_part_info) {
                distinguishable_graphs.insert(graph_part_info, graph);
            }
        }

        if distinguishable_graphs.len() >= graph_num {
            Some(distinguishable_graphs)
        } else {
            None
        }
    }

    pub fn run() {
        let (node_num, distinguishable_graphs) = (4..)
            .map(|node_num| (node_num, create_distinguishable_graphs(node_num)))
            .find_map(|(node_num, distinguishable_graphs)| {
                if let Some(distinguishable_graphs) = distinguishable_graphs {
                    Some((node_num, distinguishable_graphs))
                } else {
                    None
                }
            })
            .unwrap();

        assert_eq!(distinguishable_graphs.len(), get_graph_num());

        println!("{}", node_num);
        for (_, graph) in &distinguishable_graphs {
            graph.show_edges()
        }
        io::stdout().flush().unwrap();

        let graph_part_info_list = distinguishable_graphs
            .into_iter()
            .map(|(graph_part_info, _)| graph_part_info)
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let received_graph = Graph::from_chars(&hh, node_num);
            let received_graph_part_info = GraphPartInfo::new(&received_graph);

            let predicted_graph_idx = graph_part_info_list
                .iter()
                .find_position(|&graph_part_info| graph_part_info == &received_graph_part_info)
                .unwrap()
                .0;

            println!("{}", predicted_graph_idx);
            io::stdout().flush().unwrap();
        }
    }

    pub fn run2() {
        let graph_num = get_graph_num();

        let node_num = if graph_num <= 11 {
            4
        } else if graph_num <= 34 {
            5
        } else {
            6
        };

        let (labeled_graphs, unique_graphs) = Graph::create_labeled_graphs(node_num);

        println!("{}", node_num);
        for graph in unique_graphs.iter().take(graph_num) {
            graph.show_edges();
        }

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let received_graph = Graph::from_chars(&hh, node_num);

            let ans = *labeled_graphs.get(&received_graph).unwrap();
            println!("{}", ans);
        }
    }
}

pub mod with_noise {
    use std::io::{self, Write};

    use itertools::Itertools;

    use crate::{
        binary_search::{binary_search_with_f64, binary_search_with_usize},
        calc_edge_domain, get_epsilon, get_graph_num,
        graph::{Graph, GraphPartInfo},
        MAX_NODE_NUM, MIN_NODE_NUM, NODE_NUM_RANGE, QUERY_NUM,
    };

    pub fn create_graphs(node_num: usize, expected_interval: f64) -> (Vec<Graph>, bool) {
        assert!(MIN_NODE_NUM <= node_num && node_num <= MAX_NODE_NUM);

        let graph_num = get_graph_num();
        let edge_domain = calc_edge_domain(node_num);
        let epsilon = get_epsilon();

        let mut graphs = vec![Graph::new(node_num)];
        let mut satisfied = true;

        for graph_idx in 1..graph_num {
            let prev_expected_edge_num = graphs[graph_idx - 1].expected_edge_num();

            let is_ok = |x| {
                let expected_edge_num = x + epsilon * (edge_domain as f64 - 2.0 * x);
                expected_edge_num - prev_expected_edge_num >= expected_interval
            };

            let desired_edge_num =
                binary_search_with_f64(0.0..(edge_domain as f64), is_ok, 1e-6, false);

            let edge_num = if let Some(edge_num) = desired_edge_num {
                edge_num.ceil() as usize
            } else {
                satisfied = false;
                edge_domain
            };

            graphs.push(Graph::from_edge_num(node_num, edge_num));
        }

        (graphs, satisfied)
    }

    pub fn run2(expected_interval: f64) {
        let is_ok = |x| create_graphs(x, expected_interval).1;
        let node_num =
            binary_search_with_usize(NODE_NUM_RANGE.clone(), is_ok, false).unwrap_or(MAX_NODE_NUM);

        let graphs = create_graphs(node_num, expected_interval).0;

        println!("{}", node_num);
        graphs.iter().for_each(|graph| graph.show_edges());
        io::stdout().flush().unwrap();

        let graph_part_info_list = graphs
            .iter()
            .map(|graph| GraphPartInfo::new(graph))
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let received_graph = Graph::from_chars(&hh, node_num);
            let received_graph_part_info = GraphPartInfo::new(&received_graph);

            // let ans = graphs
            //     .iter()
            //     .position_min_by(|x, y| {
            //         (x.expected_edge_num() - received_graph.total_edge_num() as f64)
            //             .abs()
            //             .partial_cmp(
            //                 &(y.expected_edge_num() - received_graph.total_edge_num() as f64).abs(),
            //             )
            //             .unwrap()
            //     })
            //     .unwrap();

            let ans = graph_part_info_list
                .iter()
                .position_min_by(|&x, &y| {
                    x.energy_by_degrees(&received_graph_part_info)
                        .partial_cmp(&y.energy_by_degrees(&received_graph_part_info))
                        .unwrap()
                })
                .unwrap();

            println!("{}", ans);
        }
    }

    pub fn run(interval: usize) {
        let mut rng = rand::thread_rng();

        let graph_num = get_graph_num();

        let max_edge_domain = calc_edge_domain(MAX_NODE_NUM);

        let req_edge_domain = (interval * (graph_num - 1)).min(max_edge_domain);
        let node_num = NODE_NUM_RANGE
            .clone()
            .find(|&i| calc_edge_domain(i) >= req_edge_domain)
            .unwrap();
        let edge_domain = calc_edge_domain(node_num);

        let mut extra_cnt = 0;
        let graphs = (0..graph_num)
            .map(|graph_idx| {
                if interval * graph_idx <= edge_domain {
                    Graph::from_edge_num(node_num, interval * graph_idx)
                } else if edge_domain >= 2 * interval
                    && interval * (extra_cnt + 1) <= edge_domain - interval
                {
                    extra_cnt += 1;
                    Graph::from_balanced_edges(node_num, interval * extra_cnt)
                } else {
                    Graph::from_edge_num_with_random(node_num, edge_domain / 2, &mut rng)
                }
                // Graph::from_edge_num(node_num, (interval * graph_idx) % edge_domain)
                // Graph::from_edge_num_with_random(
                //     node_num,
                //     (interval * graph_idx).min(edge_domain),
                //     &mut rng,
                // )
                // if interval * graph_idx <= edge_domain {
                //     Graph::from_edge_num_with_random(node_num, interval * graph_idx, &mut rng)
                // } else {
                //     let group_num = rng.gen_range(2, node_num / 2 + 1);
                //     let group_size = node_num / group_num;

                //     let mut graph = Graph::new(node_num);

                //     for group_idx in 0..group_num {
                //         let left = group_size * group_idx;
                //         let right = group_size * (group_idx + 1);

                //         for from in left..(right - 1) {
                //             for to in (from + 1)..right {
                //                 graph.add_edge(from, to);
                //             }
                //         }
                //     }

                //     // let p = rng.gen_range(0.0, 1.0);

                //     // let mut graph = Graph::new(node_num);

                //     // let mut from = 0;
                //     // let mut to = 1;

                //     // for _ in 0..edge_domain {
                //     //     if rng.gen_bool(p) {
                //     //         graph.add_edge(from, to);
                //     //     }

                //     //     to += 1;
                //     //     if to == node_num {
                //     //         from += 1;
                //     //         to = from + 1;
                //     //     }
                //     // }

                //     graph
                // }
            })
            .collect_vec();

        // let graphs = (0..graph_num)
        //     .map(|graph_idx| {
        //         Graph::from_edge_num(node_num, (interval * graph_idx).min(edge_domain))
        //     })
        //     .collect_vec();

        println!("{}", node_num);
        graphs.iter().for_each(|graph| graph.show_edges());
        io::stdout().flush().unwrap();

        let graph_part_info_list = graphs
            .iter()
            .map(|graph| GraphPartInfo::new(graph))
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let received_graph = Graph::from_chars(&hh, node_num);
            let received_graph_part_info = GraphPartInfo::new(&received_graph);

            let ans = graph_part_info_list
                .iter()
                .position_min_by(|&x, &y| {
                    x.energy_by_degrees(&received_graph_part_info)
                        .partial_cmp(&y.energy_by_degrees(&received_graph_part_info))
                        .unwrap()
                })
                .unwrap();
            println!("{}", ans);
        }
    }
}

pub mod strategy {
    use std::io::{self, Write};

    use itertools::Itertools;

    use crate::{
        calc_edge_domain, get_graph_num, graph::Graph, MAX_NODE_NUM, MIN_NODE_NUM, NODE_NUM_RANGE,
        QUERY_NUM,
    };

    pub fn fixed_output(graph_num: usize) {
        println!("{}", MIN_NODE_NUM);
        for _ in 0..graph_num {
            println!("000000");
        }
        io::stdout().flush().unwrap();

        for _ in 0..QUERY_NUM {
            let _hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            println!("0");
            io::stdout().flush().unwrap();
        }
    }

    pub fn without_noise(graph_num: usize) {
        let node_num = NODE_NUM_RANGE
            .clone()
            .find(|&i| i * (i - 1) / 2 + 1 >= graph_num)
            .unwrap();
        let edge_domain = calc_edge_domain(node_num);

        println!("{}", node_num);
        for i in 0..graph_num {
            println!(
                "{}{}",
                String::from("1").repeat(i),
                String::from("0").repeat(edge_domain - i)
            );
        }
        io::stdout().flush().unwrap();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let edge_num = hh.iter().filter(|&&h| h == '1').count();

            let pred = if edge_num < graph_num { edge_num } else { 0 };

            println!("{}", pred);
            io::stdout().flush().unwrap();
        }
    }

    pub fn pred_by_edge_num_2(interval_rate: f64) {
        let graph_num = get_graph_num();
        let half_graph_num = (graph_num - 1) / 2;

        let max_edge_domain = calc_edge_domain(MAX_NODE_NUM);

        let calc_interval = |graph_idx| {
            (((graph_idx as f64 - half_graph_num as f64).abs() * interval_rate).round() as usize)
                .max(1)
        };

        let edge_domain = (0..graph_num)
            .map(|graph_idx| calc_interval(graph_idx))
            .sum::<usize>()
            .min(max_edge_domain);
        let node_num = NODE_NUM_RANGE
            .clone()
            .find(|&i| calc_edge_domain(i) >= edge_domain)
            .unwrap();
        let edge_domain = calc_edge_domain(node_num);
        let half_edge_domain = (edge_domain - 1) / 2;

        let mut edge_num_list = vec![0; graph_num];
        edge_num_list[half_graph_num] = half_edge_domain;

        for graph_idx in (0..half_graph_num).rev() {
            let interval = calc_interval(graph_idx);
            edge_num_list[graph_idx] = edge_num_list[graph_idx + 1].saturating_sub(interval);
        }

        for graph_idx in (half_graph_num + 1)..graph_num {
            let interval = calc_interval(graph_idx);
            edge_num_list[graph_idx] = (edge_num_list[graph_idx - 1] + interval).min(edge_domain);
        }

        let graphs = edge_num_list
            .iter()
            .map(|&edge_num| Graph::from_edge_num(node_num, edge_num))
            .collect_vec();

        println!("{}", node_num);
        graphs.iter().for_each(|graph| graph.show_edges());
        io::stdout().flush().unwrap();

        let expected_edge_num_list = graphs
            .iter()
            .map(|graph| graph.expected_edge_num())
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let edge_num = hh.iter().filter(|&&h| h == '1').count();

            let ans = expected_edge_num_list
                .iter()
                .position_min_by(|&&x, &&y| {
                    ((edge_num as f64 - x).abs())
                        .partial_cmp(&(edge_num as f64 - y).abs())
                        .unwrap()
                })
                .unwrap();

            println!("# {}", ans);
            io::stdout().flush().unwrap();
            println!("{}", ans);
            io::stdout().flush().unwrap();
        }
    }

    pub fn pred_by_edge_num(interval: usize) {
        let graph_num = get_graph_num();

        let max_edge_domain = calc_edge_domain(MAX_NODE_NUM);

        let req_edge_domain = (interval * (graph_num - 1)).min(max_edge_domain);
        let node_num = NODE_NUM_RANGE
            .clone()
            .find(|&i| calc_edge_domain(i) >= req_edge_domain)
            .unwrap();
        let edge_domain = calc_edge_domain(node_num);

        let graphs = (0..graph_num)
            .map(|graph_idx| {
                Graph::from_edge_num(node_num, (interval * graph_idx).min(edge_domain))
            })
            .collect_vec();

        println!("{}", node_num);
        graphs.iter().for_each(|graph| graph.show_edges());
        io::stdout().flush().unwrap();

        let expected_edge_num_list = graphs
            .iter()
            .map(|graph| graph.expected_edge_num())
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let edge_num = hh.iter().filter(|&&h| h == '1').count();

            let ans = expected_edge_num_list
                .iter()
                .position_min_by(|&&x, &&y| {
                    ((edge_num as f64 - x).abs())
                        .partial_cmp(&(edge_num as f64 - y).abs())
                        .unwrap()
                })
                .unwrap();

            println!("# {}", ans);
            io::stdout().flush().unwrap();
            println!("{}", ans);
            io::stdout().flush().unwrap();
        }
    }

    pub fn predict_by_connected_node_num() {
        let graph_num = get_graph_num();
        let node_num = graph_num - 1;

        let mut graphs = vec![Graph::new(node_num); graph_num];

        for graph_idx in 1..graph_num {
            for from in 0..(graph_idx - 1) {
                for to in (from + 1)..graph_idx {
                    graphs[graph_idx].add_edge(from, to);
                }
            }
        }

        println!("{}", node_num);
        for graph in &graphs {
            graph.show_edges();
        }

        let expected_edge_num_list = graphs
            .iter()
            .map(|graph| graph.expected_edge_num())
            .collect_vec();

        for _ in 0..QUERY_NUM {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let received_graph = Graph::from_chars(&hh, node_num);
            let received_total_edge_num = received_graph.total_edge_num();

            let predicted_graph_idx = expected_edge_num_list
                .iter()
                .position_min_by(|&&x, &&y| {
                    ((x - received_total_edge_num as f64).abs())
                        .partial_cmp(&(y - received_total_edge_num as f64).abs())
                        .unwrap()
                })
                .unwrap();

            println!("{}", predicted_graph_idx);
            io::stdout().flush().unwrap();
        }
    }
}

pub mod binary_search {
    //! Implements binary search for range represented by the Rust's built-in range type.

    use std::ops::{
        Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
    };

    macro_rules! impl_binary_search_with_integer {
        ($int_type: ident, $fn_name_for_inc: ident, $fn_name_for_dec: ident, $fn_name: ident, $trait_name: ident) => {
            fn $fn_name_for_inc<R, F>(rng: R, mut is_ok: F) -> Option<$int_type>
            where
                R: RangeBounds<$int_type>,
                F: FnMut($int_type) -> bool,
            {
                let mut left = match rng.start_bound() {
                    std::ops::Bound::Included(&start) => start,
                    std::ops::Bound::Excluded(&start) => {
                        if start == std::$int_type::MAX {
                            return None;
                        }

                        start + 1
                    }
                    std::ops::Bound::Unbounded => std::$int_type::MIN,
                };

                let mut right = match rng.end_bound() {
                    std::ops::Bound::Included(&end) => {
                        if end == std::$int_type::MAX {
                            if !is_ok(end) {
                                return None;
                            }

                            end
                        } else {
                            end + 1
                        }
                    }
                    std::ops::Bound::Excluded(&end) => end,
                    std::ops::Bound::Unbounded => std::$int_type::MAX,
                };

                if left >= right {
                    return None;
                }

                if is_ok(left) {
                    return Some(left);
                }

                if left + 1 == right || !is_ok(right - 1) {
                    return None;
                }

                while right - left > 1 {
                    let mid = left + (right - left) / 2;

                    if is_ok(mid) {
                        right = mid;
                    } else {
                        left = mid;
                    }
                }

                Some(right)
            }

            fn $fn_name_for_dec<R, F>(rng: R, mut is_ok: F) -> Option<$int_type>
            where
                R: RangeBounds<$int_type>,
                F: FnMut($int_type) -> bool,
            {
                let mut left = match rng.start_bound() {
                    std::ops::Bound::Included(&start) => start,
                    std::ops::Bound::Excluded(&start) => {
                        if start == std::$int_type::MAX {
                            return None;
                        }

                        start + 1
                    }
                    std::ops::Bound::Unbounded => std::$int_type::MIN,
                };

                let mut right = match rng.end_bound() {
                    std::ops::Bound::Included(&end) => {
                        if end == std::$int_type::MAX {
                            if is_ok(end) {
                                return Some(end);
                            }

                            end
                        } else {
                            end + 1
                        }
                    }
                    std::ops::Bound::Excluded(&end) => end,
                    std::ops::Bound::Unbounded => std::$int_type::MAX,
                };

                if left >= right {
                    return None;
                }

                if is_ok(right - 1) {
                    return Some(right - 1);
                }

                if left + 1 == right || !is_ok(left) {
                    return None;
                }

                while right - left > 1 {
                    let mid = left + (right - left) / 2;

                    if is_ok(mid) {
                        left = mid;
                    } else {
                        right = mid;
                    }
                }

                Some(left)
            }

            /// If `is_ok` is monotonically increasing, returns the smallest integer `x`
            /// that satisfies `is_ok(x) = true` as the value of `Some`.
            ///
            /// If `is_ok` is monotonically decreasing, returns the largest integer `x`
            /// that satisfies `is_ok(x) = true` as the value of `Some`.
            ///
            /// Returns `None` if no such integer exists in both of the above cases.
            ///
            /// # Arguments
            ///
            /// * `rng` - Domain of function `is_ok`.
            /// * `is_ok` - Monotonic function.
            /// * `dec` - Represents that `is_ok` is a monotonically decreasing function if true,
            /// or a monotonically increasing function if false.
            ///
            /// # Examples
            ///
            /// ```
            /// use atcoder8_library::binary_search::binary_search_with_i64;
            ///
            /// let is_ok = |x: i64| { x.pow(2) >= 400 };
            /// assert_eq!(binary_search_with_i64(0..100, is_ok, false), Some(20));
            ///
            /// let is_ok = |x: i64| { x.pow(2) >= 400 };
            /// assert_eq!(binary_search_with_i64(0..10, is_ok, false), None);
            ///
            /// let is_ok = |x: i64| { x.pow(3) < -8000 };
            /// assert_eq!(binary_search_with_i64(-100..0, is_ok, true), Some(-21));
            /// ```
            pub fn $fn_name<R, F>(rng: R, is_ok: F, dec: bool) -> Option<$int_type>
            where
                R: RangeBounds<$int_type>,
                F: FnMut($int_type) -> bool,
            {
                if dec {
                    $fn_name_for_dec(rng, is_ok)
                } else {
                    $fn_name_for_inc(rng, is_ok)
                }
            }

            pub trait $trait_name: Sized + RangeBounds<$int_type> {
                /// Performs a binary search on the domain specified by the Rust's built-in range type.
                ///
                /// If `is_ok` is monotonically increasing, returns the smallest integer `x`
                /// that satisfies `is_ok(x) = true` as the value of `Some`.
                ///
                /// If `is_ok` is monotonically decreasing, returns the largest integer `x`
                /// that satisfies `is_ok(x) = true` as the value of `Some`.
                ///
                /// Returns `None` if no such integer exists in both of the above cases.
                ///
                /// # Arguments
                ///
                /// * `is_ok` - Monotonic function.
                /// * `dec` - Represents that `is_ok` is a monotonically decreasing function if true,
                /// or a monotonically increasing function if false.
                ///
                /// # Examples
                ///
                /// ```
                /// use atcoder8_library::binary_search::BinarySearchWithI64;
                ///
                /// let is_ok = |x: i64| { x.pow(2) >= 400 };
                /// assert_eq!((0..100).binary_search(is_ok, false), Some(20));
                ///
                /// let is_ok = |x: i64| { x.pow(2) >= 400 };
                /// assert_eq!((0..10).binary_search(is_ok, false), None);
                ///
                /// let is_ok = |x: i64| { x.pow(3) < -8000 };
                /// assert_eq!((-100..0).binary_search(is_ok, true), Some(-21));
                /// ```
                fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<$int_type>
                where
                    F: FnMut($int_type) -> bool,
                {
                    $fn_name(self, is_ok, dec)
                }
            }

            impl $trait_name for RangeFull {}

            impl $trait_name for RangeTo<$int_type> {}

            impl $trait_name for RangeToInclusive<$int_type> {}

            impl $trait_name for RangeFrom<$int_type> {}

            impl $trait_name for Range<$int_type> {}

            impl $trait_name for RangeInclusive<$int_type> {}
        };
    }

    impl_binary_search_with_integer!(
        i8,
        binary_search_with_i8_for_inc,
        binary_search_with_i8_for_dec,
        binary_search_with_i8,
        BinarySearchWithI8
    );

    impl_binary_search_with_integer!(
        i16,
        binary_search_with_i16_for_inc,
        binary_search_with_i16_for_dec,
        binary_search_with_i16,
        BinarySearchWithI16
    );

    impl_binary_search_with_integer!(
        i32,
        binary_search_with_i32_for_inc,
        binary_search_with_i32_for_dec,
        binary_search_with_i32,
        BinarySearchWithI32
    );

    impl_binary_search_with_integer!(
        i64,
        binary_search_with_i64_for_inc,
        binary_search_with_i64_for_dec,
        binary_search_with_i64,
        BinarySearchWithI64
    );

    impl_binary_search_with_integer!(
        i128,
        binary_search_with_i128_for_inc,
        binary_search_with_i128_for_dec,
        binary_search_with_i128,
        BinarySearchWithI128
    );

    impl_binary_search_with_integer!(
        isize,
        binary_search_with_isize_for_inc,
        binary_search_with_isize_for_dec,
        binary_search_with_isize,
        BinarySearchWithIsize
    );

    impl_binary_search_with_integer!(
        u8,
        binary_search_with_u8_for_inc,
        binary_search_with_u8_for_dec,
        binary_search_with_u8,
        BinarySearchWithU8
    );

    impl_binary_search_with_integer!(
        u16,
        binary_search_with_u16_for_inc,
        binary_search_with_u16_for_dec,
        binary_search_with_u16,
        BinarySearchWithU16
    );

    impl_binary_search_with_integer!(
        u32,
        binary_search_with_u32_for_inc,
        binary_search_with_u32_for_dec,
        binary_search_with_u32,
        BinarySearchWithU32
    );

    impl_binary_search_with_integer!(
        u64,
        binary_search_with_u64_for_inc,
        binary_search_with_u64_for_dec,
        binary_search_with_u64,
        BinarySearchWithU64
    );

    impl_binary_search_with_integer!(
        u128,
        binary_search_with_u128_for_inc,
        binary_search_with_u128_for_dec,
        binary_search_with_u128,
        BinarySearchWithU128
    );

    impl_binary_search_with_integer!(
        usize,
        binary_search_with_usize_for_inc,
        binary_search_with_usize_for_dec,
        binary_search_with_usize,
        BinarySearchWithUsize
    );

    macro_rules! impl_binary_search_with_float {
        ($float_type: ident, $fn_name_for_inc: ident, $fn_name_for_dec: ident, $fn_name: ident, $trait_name: ident) => {
            fn $fn_name_for_inc<R, F>(rng: R, mut is_ok: F, eps: $float_type) -> Option<$float_type>
            where
                R: RangeBounds<$float_type>,
                F: FnMut($float_type) -> bool,
            {
                let mut left = match rng.start_bound() {
                    std::ops::Bound::Included(&start) => start,
                    std::ops::Bound::Excluded(&start) => start,
                    std::ops::Bound::Unbounded => std::$float_type::MIN,
                };

                let mut right = match rng.end_bound() {
                    std::ops::Bound::Included(&end) => end,
                    std::ops::Bound::Excluded(&end) => end,
                    std::ops::Bound::Unbounded => std::$float_type::MAX,
                };

                assert!(
                    eps > 0.0,
                    "Allowable margin of error must be a positive number."
                );

                if left >= right {
                    return None;
                }

                if is_ok(left) {
                    return Some(left);
                }

                if !is_ok(right) {
                    return None;
                }

                while right - left > eps {
                    let mid = right - (right - left) / 2.0;

                    if mid <= left || right <= mid {
                        return None;
                    }

                    if is_ok(mid) {
                        right = mid;
                    } else {
                        left = mid;
                    }
                }

                Some(right)
            }

            fn $fn_name_for_dec<R, F>(rng: R, mut is_ok: F, eps: $float_type) -> Option<$float_type>
            where
                R: RangeBounds<$float_type>,
                F: FnMut($float_type) -> bool,
            {
                let mut left = match rng.start_bound() {
                    std::ops::Bound::Included(&start) => start,
                    std::ops::Bound::Excluded(&start) => start,
                    std::ops::Bound::Unbounded => std::$float_type::MIN,
                };

                let mut right = match rng.end_bound() {
                    std::ops::Bound::Included(&end) => end,
                    std::ops::Bound::Excluded(&end) => end,
                    std::ops::Bound::Unbounded => std::$float_type::MAX,
                };

                assert!(
                    eps > 0.0,
                    "Allowable margin of error must be a positive number."
                );

                if left >= right {
                    return None;
                }

                if is_ok(right) {
                    return Some(right);
                }

                if !is_ok(left) {
                    return None;
                }

                while (right - left) > eps {
                    let mid = right - (right - left) / 2.0;

                    if mid <= left || right <= mid {
                        return None;
                    }

                    if is_ok(mid) {
                        left = mid;
                    } else {
                        right = mid;
                    }
                }

                Some(left)
            }

            /// If `is_ok` is monotonically increasing,
            /// returns the smallest floating point number `x`
            /// that satisfies `is_ok(x) = true` as the value of `Some`.
            ///
            /// If `is_ok` is monotonically decreasing,
            /// returns the largest floating point number `x`
            /// that satisfies `is_ok(x) = true` as the value of `Some`.
            ///
            /// Returns `None` if no such floating point number exists in both of the above cases.
            /// This includes the case where the absolute error cannot be determined
            /// to be less than or equal to `eps`.
            ///
            /// # Arguments
            ///
            /// * `rng` - Domain of function `is_ok`.
            /// * `is_ok` - Monotonic function.
            /// * `eps` - The allowable absolute error. It must be a positive number.
            /// * `dec` - Represents that `is_ok` is a monotonically decreasing function if true,
            /// or a monotonically increasing function if false.
            ///
            /// # Examples
            ///
            /// ```
            /// use atcoder8_library::binary_search::binary_search_with_f64;
            ///
            /// let is_ok = |x: f64| { x.powi(2) >= 400.0 };
            /// let ans = binary_search_with_f64(0.0..100.0, is_ok, 1e-6, false).unwrap();
            /// assert!((ans - 20.0).abs() <= 1e-6);
            ///
            /// let is_ok = |x: f64| { x.powi(2) >= 400.0 };
            /// assert_eq!(binary_search_with_f64(0.0..10.0, is_ok, 1e-6, false), None);
            ///
            /// let is_ok = |x: f64| { x.powi(3) <= -8000.0 };
            /// let ans = binary_search_with_f64(-100.0..0.0, is_ok, 1e-6, true).unwrap();
            /// assert!((ans - (-20.0)).abs() <= 1e-6);
            /// ```
            pub fn $fn_name<R, F>(
                rng: R,
                is_ok: F,
                eps: $float_type,
                dec: bool,
            ) -> Option<$float_type>
            where
                R: RangeBounds<$float_type>,
                F: FnMut($float_type) -> bool,
            {
                if dec {
                    $fn_name_for_dec(rng, is_ok, eps)
                } else {
                    $fn_name_for_inc(rng, is_ok, eps)
                }
            }

            pub trait $trait_name: Sized + RangeBounds<$float_type> {
                /// Performs a binary search on the domain specified by the Rust's built-in range type.
                ///
                /// If `is_ok` is monotonically increasing,
                /// returns the smallest floating point number `x`
                /// that satisfies `is_ok(x) = true` as the value of `Some`.
                ///
                /// If `is_ok` is monotonically decreasing,
                /// returns the largest floating point number `x`
                /// that satisfies `is_ok(x) = true` as the value of `Some`.
                ///
                /// Returns `None` if no such floating point number exists in both of the above cases.
                /// This includes the case where the absolute error cannot be determined
                /// to be less than or equal to `eps`.
                ///
                /// # Arguments
                ///
                /// * `is_ok` - Monotonic function.
                /// * `eps` - The allowable absolute error. It must be a positive number.
                /// * `dec` - Represents that `is_ok` is a monotonically decreasing function if true,
                /// or a monotonically increasing function if false.
                ///
                /// # Examples
                ///
                /// ```
                /// use atcoder8_library::binary_search::BinarySearchWithF64;
                ///
                /// let is_ok = |x: f64| { x.powi(2) >= 400.0 };
                /// let ans = (0.0..100.0).binary_search(is_ok, 1e-6, false).unwrap();
                /// assert!((ans - 20.0).abs() <= 1e-6);
                ///
                /// let is_ok = |x: f64| { x.powi(2) >= 400.0 };
                /// assert_eq!((0.0..10.0).binary_search(is_ok, 1e-6, false), None);
                ///
                /// let is_ok = |x: f64| { x.powi(3) <= -8000.0 };
                /// let ans = (-100.0..0.0).binary_search(is_ok, 1e-6, true).unwrap();
                /// assert!((ans - (-20.0)).abs() <= 1e-6);
                /// ```
                fn binary_search<F>(
                    self,
                    is_ok: F,
                    eps: $float_type,
                    dec: bool,
                ) -> Option<$float_type>
                where
                    F: FnMut($float_type) -> bool,
                {
                    $fn_name(self, is_ok, eps, dec)
                }
            }

            impl $trait_name for RangeFull {}

            impl $trait_name for RangeTo<$float_type> {}

            impl $trait_name for RangeToInclusive<$float_type> {}

            impl $trait_name for RangeFrom<$float_type> {}

            impl $trait_name for Range<$float_type> {}

            impl $trait_name for RangeInclusive<$float_type> {}
        };
    }

    impl_binary_search_with_float!(
        f32,
        binary_search_with_f32_for_inc,
        binary_search_with_f32_for_dec,
        binary_search_with_f32,
        BinarySearchWithF32
    );

    impl_binary_search_with_float!(
        f64,
        binary_search_with_f64_for_inc,
        binary_search_with_f64_for_dec,
        binary_search_with_f64,
        BinarySearchWithF64
    );
}
