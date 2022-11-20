// This is the 8th submission.

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

fn main() {
    let (graph_num, epsilon) = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        (
            iter.next().unwrap().parse::<usize>().unwrap(),
            iter.next().unwrap().parse::<String>().unwrap(),
        )
    };

    set_graph_num(graph_num);

    let noise_rate: usize = epsilon.split('.').skip(1).next().unwrap().parse().unwrap();

    set_noise_rate(noise_rate);

    if noise_rate == 0 {
        strategy::without_noise(graph_num);
    } else {
        strategy::pred_by_edge_num(8 * noise_rate);
    }
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

pub mod graph {
    use std::io::{self, Write};

    use crate::get_noise_rate;

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

        pub fn from_edge_num(edge_num: usize, node_num: usize) -> Self {
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

        pub fn show_graph(&self) {
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

    pub fn pred_by_edge_num(interval: usize) {
        let graph_num = get_graph_num();

        let max_edge_domain = calc_edge_domain(MAX_NODE_NUM);

        let edge_domain = (interval * (graph_num - 1)).min(max_edge_domain);
        let node_num = NODE_NUM_RANGE
            .clone()
            .find(|&i| calc_edge_domain(i) >= edge_domain)
            .unwrap();

        let graphs = (0..graph_num)
            .map(|graph_idx| {
                Graph::from_edge_num((interval * graph_idx).min(edge_domain), node_num)
            })
            .collect_vec();

        println!("{}", node_num);
        graphs.iter().for_each(|graph| graph.show_graph());
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
            graph.show_graph();
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
