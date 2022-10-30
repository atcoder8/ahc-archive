use candy_box::CANDY_TYPE_NUM;
use itertools::Itertools;
use strategy::mountain_climbing;

fn main() {
    let candy_types = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        line.split_whitespace()
            .map(|x| x.parse::<usize>().unwrap() - 1)
            .collect::<Vec<_>>()
    };

    let candy_nums = (0..CANDY_TYPE_NUM)
        .map(|i| {
            candy_types
                .iter()
                .filter(|&&candy_type| candy_type == i)
                .count()
        })
        .collect_vec();

    let denom: usize = candy_nums.iter().map(|&candy_num| candy_num.pow(2)).sum();

    let score_coef = 1e6 / denom as f64;
    candy_box::set_score_coef(score_coef);

    let finished_candy_box = mountain_climbing::mountain_climbing_1(&candy_types);
    eprintln!("Score = {}", finished_candy_box.score());
}

pub mod strategy {
    pub mod mountain_climbing {
        use crate::candy_box::{CandyBox, Dir, BOX_AREA};

        pub fn mountain_climbing_1(candy_types: &Vec<usize>) -> CandyBox {
            let mut candy_box = CandyBox::new();

            for turn in 0..BOX_AREA {
                let receive_number = {
                    let mut line = String::new();
                    std::io::stdin().read_line(&mut line).unwrap();
                    line.trim().parse::<usize>().unwrap()
                };

                candy_box.receive_candy(receive_number, candy_types[turn]);

                let next_candy_boxes = vec![
                    candy_box.tilted(Dir::Up),
                    candy_box.tilted(Dir::Down),
                    candy_box.tilted(Dir::Left),
                    candy_box.tilted(Dir::Right),
                ];

                let next_candy_box = next_candy_boxes
                    .into_iter()
                    .max_by_key(|candy_box| candy_box.score())
                    .unwrap();

                candy_box = next_candy_box;

                println!("{}", candy_box.latest_dir());
            }

            candy_box
        }
    }
}

pub mod time_measurement {
    //! This module provides the ability to measure execution time.

    use std::time::Instant;

    /// This structure provides the ability to measure execution time.
    #[derive(Debug, Hash, Clone, Copy)]
    pub struct StopWatch(Instant);

    impl Default for StopWatch {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StopWatch {
        /// Instantiate this structure and start the measurement.
        pub fn new() -> Self {
            Self(Instant::now())
        }

        /// Returns the time elapsed since this structure was instantiated (in seconds).
        pub fn elapsed_time(&self) -> f64 {
            let duration = self.0.elapsed();
            duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9
        }
    }
}

pub mod candy_box {
    use crate::atcoder8_library::union_find::UnionFind;

    pub const CANDY_TYPE_NUM: usize = 3;
    pub const BOX_AREA: usize = 100;
    pub const BOX_SIDE_LEN: usize = 10;

    static mut SCORE_COEF: Option<f64> = None;

    pub fn set_score_coef(score_coef: f64) {
        unsafe {
            debug_assert!(SCORE_COEF.is_none());

            SCORE_COEF = Some(score_coef);
        }
    }

    pub fn get_score_coef() -> f64 {
        unsafe { SCORE_COEF.unwrap() }
    }

    pub fn coord_to_idx(coord: (usize, usize)) -> usize {
        let (row, col) = coord;

        assert!(row < BOX_SIDE_LEN && col < BOX_SIDE_LEN);

        row * BOX_SIDE_LEN + col
    }

    pub fn idx_to_coord(cell_idx: usize) -> (usize, usize) {
        assert!(cell_idx < BOX_AREA);

        (cell_idx / BOX_SIDE_LEN, cell_idx % BOX_SIDE_LEN)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Dir {
        Up,
        Down,
        Left,
        Right,
    }

    #[derive(Debug, Clone)]
    pub struct CandyBox {
        candies: Vec<u128>,
        score: usize,
        turn: usize,
        latest_dir: char,
        tilted_flag: bool,
    }

    impl CandyBox {
        pub fn new() -> Self {
            Self {
                candies: vec![0; CANDY_TYPE_NUM],
                score: 0,
                turn: 0,
                latest_dir: '\0',
                tilted_flag: true,
            }
        }

        pub fn filled_by_specific_candy(&self, cell_idx: usize, candy_type: usize) -> bool {
            debug_assert!(candy_type < CANDY_TYPE_NUM);
            (self.candies[candy_type] >> cell_idx) & 1 == 1
        }

        pub fn filled(&self, cell_idx: usize) -> bool {
            (0..CANDY_TYPE_NUM)
                .any(|candy_type| self.filled_by_specific_candy(cell_idx, candy_type))
        }

        pub fn filled_candy_type(&self, cell_idx: usize) -> Option<usize> {
            (0..CANDY_TYPE_NUM)
                .position(|candy_type| self.filled_by_specific_candy(cell_idx, candy_type))
        }

        pub fn add_candy(&mut self, cell_idx: usize, candy_type: usize) {
            debug_assert!(candy_type < CANDY_TYPE_NUM);
            debug_assert!(!self.filled(cell_idx));

            self.candies[candy_type] |= 1 << cell_idx;
        }

        fn remove_candy(&mut self, cell_idx: usize) -> usize {
            let candy_type = self.filled_candy_type(cell_idx).unwrap();

            self.candies[candy_type] ^= 1 << cell_idx;

            candy_type
        }

        pub fn move_candy(&mut self, from: usize, to: usize) -> usize {
            let candy_type = self.remove_candy(from);
            self.add_candy(to, candy_type);

            candy_type
        }

        fn square_sum(&self, candy_type: usize) -> usize {
            debug_assert!(candy_type < CANDY_TYPE_NUM);

            let mut uf = UnionFind::new(BOX_AREA);

            for cell_idx in 0..BOX_AREA {
                let (r, c) = (cell_idx / BOX_SIDE_LEN, cell_idx % BOX_SIDE_LEN);

                if !self.filled_by_specific_candy(cell_idx, candy_type) {
                    continue;
                }

                if r > 0 && self.filled_by_specific_candy(cell_idx - BOX_SIDE_LEN, candy_type) {
                    uf.merge(cell_idx, cell_idx - BOX_SIDE_LEN);
                }

                if r < BOX_SIDE_LEN - 1
                    && self.filled_by_specific_candy(cell_idx + BOX_SIDE_LEN, candy_type)
                {
                    uf.merge(cell_idx, cell_idx + BOX_SIDE_LEN);
                }

                if c > 0 && self.filled_by_specific_candy(cell_idx - 1, candy_type) {
                    uf.merge(cell_idx, cell_idx - 1);
                }

                if c < BOX_SIDE_LEN - 1 && self.filled_by_specific_candy(cell_idx + 1, candy_type) {
                    uf.merge(cell_idx, cell_idx + 1);
                }
            }

            (0..BOX_AREA)
                .map(|cell_idx| {
                    if self.filled_by_specific_candy(cell_idx, candy_type)
                        && cell_idx == uf.leader(cell_idx)
                    {
                        uf.size(cell_idx).pow(2)
                    } else {
                        0
                    }
                })
                .sum()
        }

        fn calc_numer(&self) -> usize {
            (0..CANDY_TYPE_NUM)
                .map(|candy_type| self.square_sum(candy_type))
                .sum()
        }

        fn calc_score(&self) -> usize {
            (get_score_coef() * self.calc_numer() as f64).round() as usize
        }

        pub fn score(&self) -> usize {
            self.score
        }

        pub fn turn(&self) -> usize {
            self.turn
        }

        pub fn finished(&self) -> bool {
            self.turn() == BOX_AREA
        }

        pub fn tilt(&mut self, dir: Dir) {
            assert!(!self.tilted_flag);

            match dir {
                Dir::Up => {
                    for col in 0..BOX_SIDE_LEN {
                        let mut cnt = 0;

                        for row in 0..BOX_SIDE_LEN {
                            let from = coord_to_idx((row, col));

                            if self.filled(from) {
                                let to = coord_to_idx((cnt, col));

                                self.move_candy(from, to);
                                cnt += 1;
                            }
                        }
                    }

                    self.latest_dir = 'F';
                }

                Dir::Down => {
                    for col in 0..BOX_SIDE_LEN {
                        let mut cnt = 0;

                        for row in (0..BOX_SIDE_LEN).rev() {
                            let from = coord_to_idx((row, col));

                            if self.filled(from) {
                                let to = coord_to_idx((BOX_SIDE_LEN - 1 - cnt, col));

                                self.move_candy(from, to);
                                cnt += 1;
                            }
                        }
                    }

                    self.latest_dir = 'B';
                }

                Dir::Left => {
                    for row in 0..BOX_SIDE_LEN {
                        let mut cnt = 0;

                        for col in 0..BOX_SIDE_LEN {
                            let from = coord_to_idx((row, col));

                            if self.filled(from) {
                                let to = coord_to_idx((row, cnt));

                                self.move_candy(from, to);
                                cnt += 1;
                            }
                        }
                    }

                    self.latest_dir = 'L';
                }

                Dir::Right => {
                    for row in 0..BOX_SIDE_LEN {
                        let mut cnt = 0;

                        for col in (0..BOX_SIDE_LEN).rev() {
                            let from = coord_to_idx((row, col));

                            if self.filled(from) {
                                let to = coord_to_idx((row, BOX_SIDE_LEN - 1 - cnt));

                                self.move_candy(from, to);
                                cnt += 1;
                            }
                        }
                    }

                    self.latest_dir = 'R';
                }
            }

            self.score = self.calc_score();

            self.tilted_flag = true;
        }

        pub fn tilted(&self, dir: Dir) -> Self {
            let mut tilted_candy_box = self.clone();
            tilted_candy_box.tilt(dir);

            tilted_candy_box
        }

        pub fn receive_candy(&mut self, receive_number: usize, candy_type: usize) {
            assert!(candy_type < CANDY_TYPE_NUM);

            assert!(self.tilted_flag);

            let mut empty_cnt = 0;

            for cell_idx in 0..BOX_AREA {
                if !self.filled(cell_idx) {
                    empty_cnt += 1;

                    if empty_cnt == receive_number {
                        self.add_candy(cell_idx, candy_type);
                        break;
                    }
                }
            }

            self.turn += 1;

            self.tilted_flag = false;
        }

        pub fn latest_dir(&self) -> char {
            self.latest_dir
        }
    }

    impl Default for CandyBox {
        fn default() -> Self {
            CandyBox::new()
        }
    }

    pub mod cmp_candy_box {
        use super::CandyBox;

        pub struct CmpByScore(CandyBox);

        impl PartialEq for CmpByScore {
            fn eq(&self, other: &Self) -> bool {
                self.0.calc_score() == other.0.calc_score()
            }
        }

        impl Eq for CmpByScore {}

        impl PartialOrd for CmpByScore {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.0.score().partial_cmp(&other.0.calc_score())
            }
        }

        impl Ord for CmpByScore {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
    }
}

pub mod atcoder8_library {
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
}
