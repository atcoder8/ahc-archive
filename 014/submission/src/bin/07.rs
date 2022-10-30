use std::collections::{BinaryHeap, HashSet};

use rand::seq::SliceRandom;
use rect_join::{calc_draw_dist, compare::*, initialize_n_and_m, RectJoinNode, Selection};
use time_measurement::StopWatch;

const TIME_LIMIT: f64 = 4.7;

fn main() {
    let (n, m) = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        (
            iter.next().unwrap().parse::<usize>().unwrap(),
            iter.next().unwrap().parse::<usize>().unwrap(),
        )
    };
    let mut xy = Vec::new();
    for _ in 0..m {
        xy.push({
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            let mut iter = line.split_whitespace();
            (
                iter.next().unwrap().parse::<usize>().unwrap(),
                iter.next().unwrap().parse::<usize>().unwrap(),
            )
        });
    }

    initialize_n_and_m(n, m);

    let init_grid_points = RectJoinNode::new(&xy);
    eprintln!("Initial Score = {}", init_grid_points.score());

    let best_grid_points = strategy1(init_grid_points);

    best_grid_points.show_history();
    eprintln!("Score = {}", best_grid_points.score());
}

#[allow(dead_code)]
fn hill_climbing(init_node: RectJoinNode) -> RectJoinNode {
    let mut node = init_node;

    loop {
        let selections = node.all_possible_selections();
        let next_nodes: Vec<RectJoinNode> = selections
            .into_iter()
            .map(|selection| node.marked_node(selection))
            .collect();
        if let Some(best_next_node) = next_nodes.into_iter().max_by_key(|x| x.score()) {
            node = best_next_node;
        } else {
            break;
        }
    }

    node
}

#[allow(dead_code)]
fn hill_climbing_2(init_node: RectJoinNode) -> RectJoinNode {
    let mut node = init_node;

    loop {
        let mut update_flag = false;

        let selections = node.all_possible_straight_selections();
        let next_nodes: Vec<RectJoinNode> = selections
            .into_iter()
            .map(|selection| node.marked_node(selection))
            .collect();
        if let Some(best_next_node) = next_nodes.into_iter().max_by_key(|x| x.score()) {
            node = best_next_node;
            update_flag = true;
        }

        let selections = node.all_possible_diagonal_selections();
        let next_nodes: Vec<RectJoinNode> = selections
            .into_iter()
            .map(|selection| node.marked_node(selection))
            .collect();
        if let Some(best_next_node) = next_nodes.into_iter().max_by_key(|x| x.score()) {
            node = best_next_node;
            update_flag = true;
        }

        if !update_flag {
            break;
        }
    }

    node
}

#[allow(dead_code)]
fn hill_climbing_3(init_node: RectJoinNode) -> RectJoinNode {
    let mut node = init_node;

    loop {
        let mut update_flag = false;

        let selections = node.all_possible_diagonal_selections();
        let next_nodes: Vec<RectJoinNode> = selections
            .into_iter()
            .map(|selection| node.marked_node(selection))
            .collect();
        if let Some(best_next_node) = next_nodes.into_iter().max_by_key(|x| x.score()) {
            node = best_next_node;
            update_flag = true;
        }

        let selections = node.all_possible_straight_selections();
        let next_nodes: Vec<RectJoinNode> = selections
            .into_iter()
            .map(|selection| node.marked_node(selection))
            .collect();
        if let Some(best_next_node) = next_nodes.into_iter().max_by_key(|x| x.score()) {
            node = best_next_node;
            update_flag = true;
        }

        if !update_flag {
            break;
        }
    }

    node
}

#[allow(dead_code)]
fn strategy1(init_node: RectJoinNode) -> RectJoinNode {
    let stop_watch = StopWatch::new();

    let mut best_node = init_node.clone();
    let mut used = HashSet::new();
    used.insert(init_node.grid_state().clone());
    let mut heap = BinaryHeap::from(vec![CompByRevAvgLen(init_node)]);

    while let Some(CompByRevAvgLen(curr_node)) = heap.pop() {
        let mut selections = curr_node.general_possible_selections();
        selections.sort_by_key(|&selection| calc_draw_dist(selection));

        for selection in selections.into_iter().take(2) {
            let marked_node = curr_node.marked_node(selection);

            if used.contains(marked_node.grid_state()) {
                continue;
            }

            used.insert(marked_node.grid_state().clone());

            if marked_node.sum_weight() > best_node.sum_weight() {
                best_node = marked_node.clone();
            }

            heap.push(CompByRevAvgLen(marked_node));
        }

        if heap.len() >= 3000 {
            let mut temp = BinaryHeap::new();
            for _ in 0..100 {
                temp.push(heap.pop().unwrap());
            }

            heap = temp;
        }

        if stop_watch.elapsed_time() > TIME_LIMIT {
            break;
        }
    }

    hill_climbing(best_node)
}

#[allow(dead_code)]
fn strategy2(init_node: RectJoinNode) -> RectJoinNode {
    let stop_watch = StopWatch::new();

    let mut rng = rand::thread_rng();

    let mut best_node = init_node.clone();
    let mut used = HashSet::new();
    used.insert(init_node.grid_state().clone());
    let mut heap = BinaryHeap::from(vec![CompBySumWeight(init_node)]);

    while let Some(CompBySumWeight(curr_node)) = heap.pop() {
        let straight_selections = curr_node.general_possible_straight_selections();
        let diagonal_selections = curr_node.general_possible_diagonal_selections();

        let mut selections: Vec<Selection> = straight_selections
            .into_iter()
            .chain(diagonal_selections.into_iter())
            .collect();

        for _ in 0..20 {
            selections.shuffle(&mut rng);

            let mut next_node = curr_node.clone();

            for &selection in &selections {
                if next_node.selectable(selection) {
                    next_node.marking(selection);
                }
            }

            if used.contains(next_node.grid_state()) {
                continue;
            }

            used.insert(next_node.grid_state().clone());

            heap.push(CompBySumWeight(next_node));
        }

        if stop_watch.elapsed_time() > TIME_LIMIT {
            break;
        }
    }

    loop {
        let mut selections = best_node.general_possible_selections();

        if selections.is_empty() {
            break;
        }

        selections.shuffle(&mut rng);

        for selection in selections {
            best_node.try_marking(selection);
        }
    }

    best_node
}

#[allow(dead_code)]
fn strategy3(init_node: RectJoinNode) -> RectJoinNode {
    let mut rng = rand::thread_rng();

    let stop_watch = StopWatch::new();

    let mut selections = init_node.all_possible_selections();
    selections.shuffle(&mut rng);

    let best_node = selections
        .into_iter()
        .take_while(|_| stop_watch.elapsed_time() <= TIME_LIMIT * 0.9)
        .map(|selection| {
            // vec![
            //     hill_climbing(init_node.marked_node(selection)),
            //     hill_climbing_2(init_node.marked_node(selection)),
            //     hill_climbing_3(init_node.marked_node(selection)),
            // ]
            // .into_iter()
            // .max_by_key(|x| x.sum_weight())
            // .unwrap()
            hill_climbing(init_node.marked_node(selection))
        })
        .max_by_key(|x| x.sum_weight());

    if let Some(best_node) = best_node {
        debug_assert_eq!(best_node.general_possible_selections().len(), 0);

        best_node
    } else {
        init_node
    }
}

// #[allow(dead_code)]
// fn strategy4(init_node: RectJoinNode) -> RectJoinNode {
//     let mut rng = rand::thread_rng();

//     let stop_watch = StopWatch::new();

//     todo!()
// }

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

pub mod rect_join {
    use std::mem;

    use fixedbitset::FixedBitSet;
    use itertools::Itertools;

    pub type Coord = (usize, usize);

    pub type Selection = [Coord; 4];

    /// Number of vertical and horizontal grid points of a square grid
    pub static mut N: Option<usize> = None;

    /// Number of grid points marked initially
    pub static mut M: Option<usize> = None;

    static mut SCORE_UNIT: Option<f64> = None;

    pub fn initialize_n_and_m(n: usize, m: usize) {
        unsafe {
            N = Some(n);
            M = Some(m);
            SCORE_UNIT = Some(6e6 / (m as f64 * (n * n + 5) as f64))
        }
    }

    pub fn n() -> usize {
        unsafe { N.unwrap() }
    }

    pub fn side_len() -> usize {
        n() - 1
    }

    pub fn m() -> usize {
        unsafe { M.unwrap() }
    }

    fn score_unit() -> f64 {
        unsafe { SCORE_UNIT.unwrap() }
    }

    pub fn weight(coord: Coord) -> usize {
        let c = (n() - 1) / 2;
        let (x, y) = coord;

        let diff_x = if x >= c { x - c } else { c - x };
        let sq_diff_x = diff_x * diff_x;

        let diff_y = if y >= c { y - c } else { c - y };
        let sq_diff_y = diff_y * diff_y;

        sq_diff_x + sq_diff_y + 1
    }

    pub fn calc_sum_weight(marks: &FixedBitSet) -> usize {
        let c = (n() - 1) / 2;

        (0..n())
            .map(|x| {
                let diff_x = if x >= c { x - c } else { c - x };
                let sq_diff_x = diff_x * diff_x;

                (0..n())
                    .map(|y| {
                        if marks[x * n() + y] {
                            let diff_y = if y >= c { y - c } else { c - y };
                            let sq_diff_y = diff_y * diff_y;

                            sq_diff_x + sq_diff_y + 1
                        } else {
                            0
                        }
                    })
                    .sum::<usize>()
            })
            .sum()
    }

    pub fn calc_dist(coord1: Coord, coord2: Coord) -> usize {
        let (x1, y1) = coord1;
        let (x2, y2) = coord2;

        let diff_x = if x1 >= x2 { x1 - x2 } else { x2 - x1 };
        let diff_y = if y1 >= y2 { y1 - y2 } else { y2 - y1 };

        debug_assert!(diff_x == 0 || diff_y == 0 || diff_x == diff_y);

        diff_x.max(diff_y)
    }

    pub fn calc_draw_dist(selection: Selection) -> usize {
        2 * (calc_dist(selection[0], selection[1]) + calc_dist(selection[1], selection[2]))
    }

    #[derive(Debug, Default, Hash, Clone, PartialEq, Eq)]
    pub struct Marks {
        pub marks: FixedBitSet,
        pub horizontal_existences: u64,
        pub vertical_existences: u64,
        pub upward_existences: u128,
        pub downward_existences: u128,
        pub sum_weight: usize,
    }

    impl Marks {
        fn add_mark(&mut self, coord: Coord) {
            let (x, y) = coord;

            debug_assert!(x < n() && y < n());
            debug_assert!(!self.marks[x * n() + y]);

            self.marks.insert(x * n() + y);

            self.vertical_existences |= 1 << x;
            self.horizontal_existences |= 1 << y;
            self.upward_existences |= 1 << (n() - 1 + x - y);
            self.downward_existences |= 1 << (x + y);

            self.sum_weight += weight(coord);
        }

        fn new(init_marked_coords: &Vec<Coord>) -> Self {
            let mut marks = Marks::default();
            marks.marks = FixedBitSet::with_capacity(n() * n());

            for &coord in init_marked_coords {
                marks.add_mark(coord);
            }

            marks
        }

        pub fn is_marked(&self, coord: Coord) -> bool {
            self.marks[coord.0 * n() + coord.1]
        }

        pub fn sum_weight(&self) -> usize {
            self.sum_weight
        }

        pub fn score(&self) -> usize {
            (score_unit() * self.sum_weight() as f64).round() as usize
        }

        fn all_vertical_line_is_empty(&self, x: usize) -> bool {
            debug_assert!(x < n());

            (self.vertical_existences >> x) & 1 == 0
        }

        fn all_horizontal_line_is_empty(&self, y: usize) -> bool {
            debug_assert!(y < n());

            (self.horizontal_existences >> y) & 1 == 0
        }

        fn all_upward_line_is_empty(&self, coord: Coord) -> bool {
            let (x, y) = coord;

            debug_assert!(x < n() && y < n());

            (self.upward_existences >> (n() - 1 + x - y)) & 1 == 0
        }

        fn all_downward_line_is_empty(&self, coord: Coord) -> bool {
            let (x, y) = coord;

            debug_assert!(x < n() && y < n());

            (self.downward_existences >> (x + y)) & 1 == 0
        }

        fn vertical_line_is_empty(&self, x: usize, y_pair: (usize, usize)) -> bool {
            let (mut y1, mut y2) = y_pair;

            debug_assert_ne!(y1, y2);

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            self.all_vertical_line_is_empty(x) || ((y1 + 1)..y2).all(|y| !self.is_marked((x, y)))
        }

        fn horizontal_line_is_empty(&self, x_pair: (usize, usize), y: usize) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            self.all_horizontal_line_is_empty(y) || ((x1 + 1)..x2).all(|x| !self.is_marked((x, y)))
        }

        #[allow(dead_code)]
        fn straight_line_is_empty(&self, coord1: Coord, coord2: Coord) -> bool {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        fn upward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            self.all_upward_line_is_empty(source_coord)
                || (1..dist).all(|i| !self.is_marked((sx + i, sy + i)))
        }

        fn downward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            self.all_downward_line_is_empty(source_coord)
                || (1..dist).all(|i| !self.is_marked((sx + i, sy - i)))
        }

        #[allow(dead_code)]
        fn diagonal_line_is_empty(&self, mut coord1: Coord, mut coord2: Coord) -> bool {
            debug_assert_ne!(coord1, coord2);

            if coord1.0 > coord2.0 {
                mem::swap(&mut coord1, &mut coord2)
            }

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            let dist = x2 - x1;

            if y1 < y2 {
                debug_assert_eq!(y2 - y1, dist);

                self.upward_line_is_empty(coord1, dist)
            } else {
                debug_assert_eq!(y1 - y2, dist);

                self.downward_line_is_empty(coord1, dist)
            }
        }

        #[allow(dead_code)]
        fn line_is_empty(&self, coord1: Coord, coord2: Coord) -> bool {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                self.diagonal_line_is_empty(coord1, coord2)
            }
        }
    }

    #[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
    pub enum Dir8 {
        Left,
        Right,
        Lower,
        Upper,
        LowerLeft,
        UpperRight,
        LowerRight,
        UpperLeft,
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct GridLines {
        pub horizontal: FixedBitSet,
        pub vertical: FixedBitSet,
        pub upward: FixedBitSet,
        pub downward: FixedBitSet,
        pub straight_drawn_len: usize,
        pub diagonal_drawn_len: usize,
    }

    impl Default for GridLines {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GridLines {
        fn new() -> Self {
            Self {
                horizontal: FixedBitSet::with_capacity(side_len() * n()),
                vertical: FixedBitSet::with_capacity(n() * side_len()),
                upward: FixedBitSet::with_capacity(side_len() * side_len()),
                downward: FixedBitSet::with_capacity(side_len() * side_len()),
                straight_drawn_len: 0,
                diagonal_drawn_len: 0,
            }
        }

        fn is_drawn(&self, source_coord: Coord, dir: Dir8) -> bool {
            let (sx, sy) = source_coord;

            match dir {
                Dir8::Left => {
                    debug_assert!(0 < sx && sx < n() && sy < n());

                    self.horizontal[(sx - 1) * n() + sy]
                }
                Dir8::Right => {
                    debug_assert!(sx < side_len() && sy < n());

                    self.horizontal[sx * n() + sy]
                }
                Dir8::Lower => {
                    debug_assert!(sx < n() && 0 < sy && sy < n());

                    self.vertical[sx * side_len() + sy - 1]
                }
                Dir8::Upper => {
                    debug_assert!(sx < n() && sy < side_len());

                    self.vertical[sx * side_len() + sy]
                }
                Dir8::LowerLeft => {
                    debug_assert!(0 < sx && sx < n() && 0 < sy && sy < n());

                    self.upward[(sx - 1) * side_len() + sy - 1]
                }
                Dir8::UpperRight => {
                    debug_assert!(sx < side_len() && sy < side_len());

                    self.upward[sx * side_len() + sy]
                }
                Dir8::LowerRight => {
                    debug_assert!(sx < side_len() && 0 < sy && sy < n());

                    self.downward[sx * side_len() + sy - 1]
                }
                Dir8::UpperLeft => {
                    debug_assert!(0 < sx && sx < n() && sy < side_len());

                    self.downward[(sx - 1) * side_len() + sy]
                }
            }
        }

        fn draw_line(&mut self, source_coord: Coord, dir: Dir8) {
            let (sx, sy) = source_coord;

            match dir {
                Dir8::Left => {
                    debug_assert!(0 < sx && sx < n() && sy < n());
                    debug_assert!(!self.horizontal[(sx - 1) * n() + sy]);

                    self.horizontal.insert((sx - 1) * n() + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::Right => {
                    debug_assert!(sx < side_len() && sy < n());
                    debug_assert!(!self.horizontal[sx * n() + sy]);

                    self.horizontal.insert(sx * n() + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::Lower => {
                    debug_assert!(sx < n() && 0 < sy && sy < n());
                    debug_assert!(!self.vertical[sx * side_len() + sy - 1]);

                    self.vertical.insert(sx * side_len() + sy - 1);
                    self.straight_drawn_len += 1;
                }
                Dir8::Upper => {
                    debug_assert!(sx < n() && sy < side_len());
                    debug_assert!(!self.vertical[sx * side_len() + sy]);

                    self.vertical.insert(sx * side_len() + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::LowerLeft => {
                    debug_assert!(0 < sx && sx < n() && 0 < sy && sy < n());
                    debug_assert!(!self.upward[(sx - 1) * side_len() + sy - 1]);

                    self.upward.insert((sx - 1) * side_len() + sy - 1);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::UpperRight => {
                    debug_assert!(sx < side_len() && sy < side_len());
                    debug_assert!(!self.upward[sx * side_len() + sy]);

                    self.upward.insert(sx * side_len() + sy);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::LowerRight => {
                    debug_assert!(sx < side_len() && 0 < sy && sy < n());
                    debug_assert!(!self.downward[sx * side_len() + sy - 1]);

                    self.downward.insert(sx * side_len() + sy - 1);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::UpperLeft => {
                    debug_assert!(0 < sx && sx < n() && sy < side_len());
                    debug_assert!(!self.downward[(sx - 1) * side_len() + sy]);

                    self.downward.insert((sx - 1) * side_len() + sy);
                    self.diagonal_drawn_len += 1;
                }
            }
        }

        fn vertical_line_is_empty(&self, x: usize, y_pair: (usize, usize)) -> bool {
            let (mut y1, mut y2) = y_pair;

            debug_assert_ne!(y1, y2);

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            (y1..y2).all(|y| !self.is_drawn((x, y), Dir8::Upper))
        }

        fn horizontal_line_is_empty(&self, x_pair: (usize, usize), y: usize) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            (x1..x2).all(|x| !self.is_drawn((x, y), Dir8::Right))
        }

        fn upward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            (0..dist).all(|i| !self.is_drawn((sx + i, sy + i), Dir8::UpperRight))
        }

        fn downward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            (0..dist).all(|i| !self.is_drawn((sx + i, sy - i), Dir8::LowerRight))
        }

        fn straight_drawn_len(&self) -> usize {
            self.straight_drawn_len
        }

        fn diagonal_drawn_len(&self) -> usize {
            self.diagonal_drawn_len
        }

        fn drawn_len(&self) -> usize {
            self.straight_drawn_len + self.diagonal_drawn_len
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct GridState {
        pub marks: Marks,
        pub lines: GridLines,
    }

    impl GridState {
        fn new(init_marked_coords: &Vec<(usize, usize)>) -> Self {
            let mut marks = FixedBitSet::with_capacity(n() * n());

            for &(x, y) in init_marked_coords {
                marks.insert(x * n() + y);
            }

            Self {
                marks: Marks::new(init_marked_coords),
                lines: GridLines::new(),
            }
        }

        pub fn is_marked(&self, coord: Coord) -> bool {
            self.marks.is_marked(coord)
        }

        pub fn sum_weight(&self) -> usize {
            self.marks.sum_weight()
        }

        pub fn score(&self) -> usize {
            self.marks.score()
        }

        pub fn vertical_line_is_empty(&self, x: usize, y_pair: (usize, usize)) -> bool {
            let (mut y1, mut y2) = y_pair;

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            self.marks.vertical_line_is_empty(x, y_pair)
                && self.lines.vertical_line_is_empty(x, y_pair)
        }

        pub fn horizontal_line_is_empty(&self, x_pair: (usize, usize), y: usize) -> bool {
            let (mut x1, mut x2) = x_pair;

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            self.marks.horizontal_line_is_empty(x_pair, y)
                && self.lines.horizontal_line_is_empty(x_pair, y)
        }

        pub fn straight_line_is_empty(&self, coord1: Coord, coord2: Coord) -> bool {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        pub fn upward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            self.marks.upward_line_is_empty(source_coord, dist)
                && self.lines.upward_line_is_empty(source_coord, dist)
        }

        pub fn downward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            self.marks.downward_line_is_empty(source_coord, dist)
                && self.lines.downward_line_is_empty(source_coord, dist)
        }

        pub fn diagonal_line_is_empty(&self, mut coord1: Coord, mut coord2: Coord) -> bool {
            debug_assert_ne!(coord1, coord2);

            if coord1.0 > coord2.0 {
                mem::swap(&mut coord1, &mut coord2)
            }

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            let dist = x2 - x1;

            if y1 < y2 {
                debug_assert_eq!(y2 - y1, dist);

                self.upward_line_is_empty(coord1, dist)
            } else {
                debug_assert_eq!(y1 - y2, dist);

                self.downward_line_is_empty(coord1, dist)
            }
        }

        pub fn line_is_empty(&self, coord1: Coord, coord2: Coord) -> bool {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                self.diagonal_line_is_empty(coord1, coord2)
            }
        }

        // Searches for the marked left x-coordinate closest to the target coordinate.
        fn search_closest_left(&self, coord: Coord) -> Option<usize> {
            let (tx, ty) = coord;

            if self.marks.all_horizontal_line_is_empty(ty) {
                return None;
            }

            (0..tx)
                .rev()
                .take_while(|&x| !self.lines.is_drawn((x, ty), Dir8::Right))
                .find(|&x| self.is_marked((x, ty)))
        }

        // Searches for the marked right x-coordinate closest to the target coordinate.
        fn search_closest_right(&self, coord: Coord) -> Option<usize> {
            let (tx, ty) = coord;

            if self.marks.all_horizontal_line_is_empty(ty) {
                return None;
            }

            ((tx + 1)..n())
                .take_while(|&x| !self.lines.is_drawn((x, ty), Dir8::Left))
                .find(|&x| self.is_marked((x, ty)))
        }

        // Searches for the marked lower y-coordinate closest to the target coordinate.
        fn search_closest_lower(&self, coord: Coord) -> Option<usize> {
            let (tx, ty) = coord;

            if self.marks.all_vertical_line_is_empty(tx) {
                return None;
            }

            (0..ty)
                .rev()
                .take_while(|&y| !self.lines.is_drawn((tx, y), Dir8::Upper))
                .find(|&y| self.is_marked((tx, y)))
        }

        // Searches for the marked upper y-coordinate closest to the target coordinate.
        fn search_closest_upper(&self, coord: Coord) -> Option<usize> {
            let (tx, ty) = coord;

            if self.marks.all_vertical_line_is_empty(tx) {
                return None;
            }

            ((ty + 1)..n())
                .take_while(|&y| !self.lines.is_drawn((tx, y), Dir8::Lower))
                .find(|&y| self.is_marked((tx, y)))
        }

        // Searches for the marked upper-right coordinate closest to the `coord`.
        fn search_closest_upper_right(&self, coord: Coord) -> Option<Coord> {
            if self.marks.all_upward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord;

            // Diagonal distance of the marked upper-right coordinate closest to the `coord`.
            let upper_right_dist = (1..=(n() - 1 - x).min(n() - 1 - y))
                .take_while(|&i| !self.lines.is_drawn((x + i, y + i), Dir8::LowerLeft))
                .find(|&i| self.is_marked((x + i, y + i)));

            upper_right_dist.map(|dist| (x + dist, y + dist))
        }

        // Searches for the marked lower-left coordinate closest to the `coord`.
        fn search_closest_lower_left(&self, coord: Coord) -> Option<Coord> {
            if self.marks.all_upward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord;

            // Diagonal distance of the marked lower-left coordinate closest to the `coord`.
            let lower_left_dist = (1..=x.min(y))
                .take_while(|&i| !self.lines.is_drawn((x - i, y - i), Dir8::UpperRight))
                .find(|&i| self.is_marked((x - i, y - i)));

            lower_left_dist.map(|dist| (x - dist, y - dist))
        }

        // Searches for the marked upper-left coordinate closest to the `coord`.
        fn search_closest_upper_left(&self, coord: Coord) -> Option<Coord> {
            if self.marks.all_downward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord;

            // Diagonal distance of the marked upper-left coordinate closest to the `coord`.
            let upper_left_dist = (1..=x.min(n() - 1 - y))
                .take_while(|&i| !self.lines.is_drawn((x - i, y + i), Dir8::LowerRight))
                .find(|&i| self.is_marked((x - i, y + i)));

            upper_left_dist.map(|dist| (x - dist, y + dist))
        }

        // Searches for the marked lower_right coordinate closest to the `coord`.
        fn search_closest_lower_right(&self, coord: Coord) -> Option<Coord> {
            if self.marks.all_downward_line_is_empty(coord) {
                return None;
            }

            let (tx, ty) = coord;

            // Diagonal distance of the marked lower-right coordinate closest to the `coord`.
            let lower_right_dist = (1..=(n() - 1 - tx).min(ty))
                .take_while(|&i| !self.lines.is_drawn((tx + i, ty - i), Dir8::UpperLeft))
                .find(|&dist| self.is_marked((tx + dist, ty - dist)));

            lower_right_dist.map(|dist| (tx + dist, ty - dist))
        }

        fn search_one_straight_selection(&self, coord: Coord) -> Option<Selection> {
            if self.is_marked(coord) {
                return None;
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            if self.marks.all_vertical_line_is_empty(tx)
                || self.marks.all_horizontal_line_is_empty(ty)
            {
                return None;
            }

            // x-coordinate of the marked left closest to the target coordinate.
            let left_x = self.search_closest_left(coord);

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = self.search_closest_right(coord);

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = self.search_closest_lower(coord);

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = self.search_closest_upper(coord);

            for (&other_x, &other_y) in [left_x, right_x]
                .iter()
                .cartesian_product([upper_y, lower_y].iter())
            {
                if let (Some(other_x), Some(other_y)) = (other_x, other_y) {
                    if self.is_marked((other_x, other_y))
                        && self.horizontal_line_is_empty((tx, other_x), other_y)
                        && self.vertical_line_is_empty(other_x, (ty, other_y))
                    {
                        return Some([coord, (other_x, ty), (other_x, other_y), (tx, other_y)]);
                    }
                }
            }

            None
        }

        fn search_one_diagonal_selection(&self, coord: Coord) -> Option<Selection> {
            if self.is_marked(coord) {
                return None;
            }

            if self.marks.all_upward_line_is_empty(coord)
                || self.marks.all_downward_line_is_empty(coord)
            {
                return None;
            }

            // The upper-right coordinate of the marked grid point is closest to the `coord`.
            let upper_right_coord = self.search_closest_upper_right(coord);

            // The lower-left coordinate of the marked grid point is closest to the `coord`.
            let lower_left_coord = self.search_closest_lower_left(coord);

            // The upper-left coordinate of the marked grid point is closest to the `coord`.
            let upper_left_coord = self.search_closest_upper_left(coord);

            // The lower-right coordinate of the marked grid point is closest to the `coord`.
            let lower_right_coord = self.search_closest_lower_right(coord);

            let diagonal_coords = [
                upper_right_coord,
                upper_left_coord,
                lower_left_coord,
                lower_right_coord,
            ];

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            for i in 0..diagonal_coords.len() {
                if let (Some(other_coord_1), Some(other_coord_2)) = (
                    diagonal_coords[i],
                    diagonal_coords[(i + 1) % diagonal_coords.len()],
                ) {
                    let (ox1, oy1) = other_coord_1;
                    let (ox2, oy2) = other_coord_2;

                    let opp_coord = if let (Some(opp_x), Some(opp_y)) =
                        ((ox1 + ox2).checked_sub(tx), (oy1 + oy2).checked_sub(ty))
                    {
                        if opp_x < n() && opp_y < n() {
                            (opp_x, opp_y)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    };

                    if self.is_marked(opp_coord)
                        && self.diagonal_line_is_empty(other_coord_1, opp_coord)
                        && self.diagonal_line_is_empty(other_coord_2, opp_coord)
                    {
                        return Some([coord, other_coord_1, opp_coord, other_coord_2]);
                    }
                }
            }

            // Returns `None` if the target coordinates cannot be marked.
            None
        }

        fn search_one_selection(&self, coord: Coord) -> Option<Selection> {
            if let Some(selection) = self.search_one_straight_selection(coord) {
                return Some(selection);
            }

            if let Some(selection) = self.search_one_diagonal_selection(coord) {
                return Some(selection);
            }

            None
        }

        fn search_straight_selections(&self, coord: Coord) -> Vec<Selection> {
            if self.is_marked(coord) {
                return vec![];
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            if self.marks.all_vertical_line_is_empty(tx)
                || self.marks.all_horizontal_line_is_empty(ty)
            {
                return vec![];
            }

            // x-coordinate of the marked left closest to the target coordinate.
            let left_x = self.search_closest_left(coord);

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = self.search_closest_right(coord);

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = self.search_closest_lower(coord);

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = self.search_closest_upper(coord);

            [left_x, right_x]
                .iter()
                .cartesian_product([upper_y, lower_y].iter())
                .filter_map(|(&other_x, &other_y)| {
                    if let (Some(other_x), Some(other_y)) = (other_x, other_y) {
                        if self.is_marked((other_x, other_y))
                            && self.horizontal_line_is_empty((tx, other_x), other_y)
                            && self.vertical_line_is_empty(other_x, (ty, other_y))
                        {
                            return Some([coord, (other_x, ty), (other_x, other_y), (tx, other_y)]);
                        }
                    }

                    None
                })
                .collect()
        }

        fn search_diagonal_selections(&self, coord: Coord) -> Vec<Selection> {
            if self.is_marked(coord) {
                return vec![];
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            if self.marks.all_upward_line_is_empty(coord)
                || self.marks.all_downward_line_is_empty(coord)
            {
                return vec![];
            }

            // The upper-right coordinate of the marked grid point is closest to the `coord`.
            let upper_right_coord = self.search_closest_upper_right(coord);

            // The lower-left coordinate of the marked grid point is closest to the `coord`.
            let lower_left_coord = self.search_closest_lower_left(coord);

            // The upper-left coordinate of the marked grid point is closest to the `coord`.
            let upper_left_coord = self.search_closest_upper_left(coord);

            // The lower-right coordinate of the marked grid point is closest to the `coord`.
            let lower_right_coord = self.search_closest_lower_right(coord);

            let diagonal_coords = [
                upper_right_coord,
                upper_left_coord,
                lower_left_coord,
                lower_right_coord,
            ];

            (0..diagonal_coords.len())
                .filter_map(|i| {
                    if let (Some(other_coord_1), Some(other_coord_2)) = (
                        diagonal_coords[i],
                        diagonal_coords[(i + 1) % diagonal_coords.len()],
                    ) {
                        let (ox1, oy1) = other_coord_1;
                        let (ox2, oy2) = other_coord_2;

                        let opp_coord = if let (Some(opp_x), Some(opp_y)) =
                            ((ox1 + ox2).checked_sub(tx), (oy1 + oy2).checked_sub(ty))
                        {
                            if opp_x < n() && opp_y < n() {
                                (opp_x, opp_y)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        };

                        if self.is_marked(opp_coord)
                            && self.diagonal_line_is_empty(other_coord_1, opp_coord)
                            && self.diagonal_line_is_empty(other_coord_2, opp_coord)
                        {
                            return Some([coord, other_coord_1, opp_coord, other_coord_2]);
                        }
                    }

                    None
                })
                .collect()
        }

        pub fn search_selections(&self, coord: Coord) -> Vec<Selection> {
            let mut straight_selections = self.search_straight_selections(coord);
            let mut diagonal_selections = self.search_diagonal_selections(coord);
            straight_selections.append(&mut diagonal_selections);

            straight_selections
        }

        fn general_possible_straight_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| {
                    (0..n()).filter_map(move |y| self.search_one_straight_selection((x, y)))
                })
                .collect()
        }

        fn general_possible_diagonal_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| {
                    (0..n()).filter_map(move |y| self.search_one_diagonal_selection((x, y)))
                })
                .collect()
        }

        fn general_possible_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| (0..n()).filter_map(move |y| self.search_one_selection((x, y))))
                .collect()
        }

        fn all_possible_straight_selections(&self) -> Vec<Selection> {
            (0..n().pow(2))
                .map(|coord_idx| {
                    self.search_straight_selections((coord_idx / n(), coord_idx % n()))
                })
                .flatten()
                .collect()
        }

        fn all_possible_diagonal_selections(&self) -> Vec<Selection> {
            (0..n().pow(2))
                .map(|coord_idx| {
                    self.search_diagonal_selections((coord_idx / n(), coord_idx % n()))
                })
                .flatten()
                .collect()
        }

        fn all_possible_selections(&self) -> Vec<Selection> {
            let mut straight_selections = self.all_possible_straight_selections();
            let mut diagonal_selections = self.all_possible_diagonal_selections();
            straight_selections.append(&mut diagonal_selections);

            straight_selections
        }

        fn draw_vertical_grid_line(&mut self, x: usize, y_pair: (usize, usize)) {
            let (mut y1, mut y2) = y_pair;
            if y1 > y2 {
                mem::swap(&mut y1, &mut y2)
            }

            for y in y1..y2 {
                self.lines.draw_line((x, y), Dir8::Upper);
            }
        }

        fn draw_horizontal_grid_line(&mut self, x_pair: (usize, usize), y: usize) {
            let (mut x1, mut x2) = x_pair;
            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            for x in x1..x2 {
                self.lines.draw_line((x, y), Dir8::Right);
            }
        }

        #[allow(dead_code)]
        fn draw_straight_grid_line(&mut self, coord1: Coord, coord2: Coord) {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.draw_vertical_grid_line(x1, (y1, y2));
            } else if y1 == y2 {
                self.draw_horizontal_grid_line((x1, x2), y1);
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        fn draw_upward_grid_line(&mut self, source_coord: Coord, dist: usize) {
            let (sx, sy) = source_coord;

            for i in 0..dist {
                self.lines.draw_line((sx + i, sy + i), Dir8::UpperRight);
            }
        }

        fn draw_downward_grid_line(&mut self, source_coord: Coord, dist: usize) {
            let (sx, sy) = source_coord;

            for i in 0..dist {
                self.lines.draw_line((sx + i, sy - i), Dir8::LowerRight);
            }
        }

        fn draw_diagonal_grid_line(&mut self, mut coord1: Coord, mut coord2: Coord) {
            if coord1.0 > coord2.0 {
                mem::swap(&mut coord1, &mut coord2)
            }

            let dist = coord2.0 - coord1.0;

            if coord1.1 < coord2.1 {
                debug_assert_eq!(coord2.1 - coord1.1, dist);

                self.draw_upward_grid_line(coord1, dist);
            } else {
                debug_assert_eq!(coord1.1 - coord2.1, dist);

                self.draw_downward_grid_line(coord1, dist);
            }
        }

        fn draw_grid_line(&mut self, coord1: Coord, coord2: Coord) {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                self.draw_vertical_grid_line(x1, (y1, y2));
            } else if y1 == y2 {
                self.draw_horizontal_grid_line((x1, x2), y1);
            } else {
                self.draw_diagonal_grid_line(coord1, coord2);
            }
        }

        fn selectable(&self, selection: Selection) -> bool {
            !self.marks.is_marked(selection[0])
                && selection
                    .iter()
                    .skip(1)
                    .all(|&coord| self.marks.is_marked(coord))
                && (0..selection.len())
                    .all(|i| self.line_is_empty(selection[i], selection[(i + 1) % selection.len()]))
        }

        fn marking(&mut self, selection: Selection) {
            debug_assert!(self.selectable(selection));

            self.marks.add_mark(selection[0]);

            for i in 0..selection.len() {
                self.draw_grid_line(selection[i], selection[(i + 1) % selection.len()]);
            }
        }

        pub fn straight_drawn_len(&self) -> usize {
            self.lines.straight_drawn_len()
        }

        pub fn diagonal_drawn_len(&self) -> usize {
            self.lines.diagonal_drawn_len()
        }

        pub fn drawn_len(&self) -> usize {
            self.lines.drawn_len()
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct RectJoinNode {
        grid_state: GridState,
        history: Vec<Selection>,
    }

    impl RectJoinNode {
        pub fn new(xy: &Vec<(usize, usize)>) -> Self {
            Self {
                grid_state: GridState::new(xy),
                history: vec![],
            }
        }

        pub fn is_marked(&self, coord: Coord) -> bool {
            self.grid_state.is_marked(coord)
        }

        pub fn sum_weight(&self) -> usize {
            self.grid_state.sum_weight()
        }

        pub fn score(&self) -> usize {
            self.grid_state.score()
        }

        pub fn general_possible_straight_selections(&self) -> Vec<Selection> {
            self.grid_state.general_possible_straight_selections()
        }

        pub fn general_possible_diagonal_selections(&self) -> Vec<Selection> {
            self.grid_state.general_possible_diagonal_selections()
        }

        pub fn general_possible_selections(&self) -> Vec<Selection> {
            self.grid_state.general_possible_selections()
        }

        pub fn all_possible_straight_selections(&self) -> Vec<Selection> {
            self.grid_state.all_possible_straight_selections()
        }

        pub fn all_possible_diagonal_selections(&self) -> Vec<Selection> {
            self.grid_state.all_possible_diagonal_selections()
        }

        pub fn all_possible_selections(&self) -> Vec<Selection> {
            self.grid_state.all_possible_selections()
        }

        pub fn selectable(&self, selection: Selection) -> bool {
            self.grid_state.selectable(selection)
        }

        pub fn marking(&mut self, selection: Selection) {
            self.grid_state.marking(selection);
            self.history.push(selection);
        }

        pub fn try_marking(&mut self, selection: Selection) -> bool {
            if !self.selectable(selection) {
                return false;
            }

            self.marking(selection);

            true
        }

        pub fn marked_node(&self, selection: Selection) -> Self {
            let mut marked_grid_points = self.clone();
            marked_grid_points.marking(selection);

            marked_grid_points
        }

        pub fn history(&self) -> &Vec<Selection> {
            &self.history
        }

        pub fn show_history(&self) {
            println!("{}", self.history.len());

            for selection in &self.history {
                let (x1, y1) = selection[0];
                let (x2, y2) = selection[1];
                let (x3, y3) = selection[2];
                let (x4, y4) = selection[3];

                println!("{} {} {} {} {} {} {} {}", x1, y1, x2, y2, x3, y3, x4, y4);
            }
        }

        pub fn moved_num(&self) -> usize {
            self.history.len()
        }

        pub fn grid_state(&self) -> &GridState {
            &self.grid_state
        }

        pub fn straight_drawn_len(&self) -> usize {
            self.grid_state.straight_drawn_len()
        }

        pub fn diagonal_drawn_len(&self) -> usize {
            self.grid_state.diagonal_drawn_len()
        }

        pub fn drawn_len(&self) -> usize {
            self.grid_state.drawn_len()
        }
    }

    pub mod compare {
        use std::cmp::Ordering;

        use super::RectJoinNode;

        #[derive(Debug, Clone)]
        pub struct CompBySumWeight(pub RectJoinNode);

        impl CompBySumWeight {
            #[allow(dead_code)]
            fn cmp_by_sum_weight(&self, other: &Self) -> Ordering {
                self.0.sum_weight().cmp(&other.0.sum_weight())
            }

            // #[allow(dead_code)]
            // pub fn cmp_by_sum_weight_and_moved_num(&self, other: &Self) -> Ordering {
            //     match self.0.sum_weight().cmp(&other.0.sum_weight()) {
            //         Ordering::Less => Ordering::Less,
            //         Ordering::Equal => self.0.moved_num().cmp(&other.0.moved_num()).reverse(),
            //         Ordering::Greater => Ordering::Greater,
            //     }
            // }

            // #[allow(dead_code)]
            // pub fn cmp_by_moved_num(&self, other: &Self) -> Ordering {
            //     self.0.moved_num().cmp(&other.0.moved_num())
            // }
        }

        impl PartialEq for CompBySumWeight {
            fn eq(&self, other: &Self) -> bool {
                self.0.sum_weight() == other.0.sum_weight()
            }
        }

        impl Eq for CompBySumWeight {}

        impl PartialOrd for CompBySumWeight {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.0.sum_weight().cmp(&other.0.sum_weight()))
            }
        }

        impl Ord for CompBySumWeight {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        #[derive(Debug, Clone)]
        pub struct CompByRevAvgLen(pub RectJoinNode);

        impl CompByRevAvgLen {
            #[allow(dead_code)]
            fn cmp_node(&self, other: &Self) -> Ordering {
                let self_avg_len = self.0.drawn_len() as f64 / self.0.moved_num() as f64;
                let other_avg_len = other.0.drawn_len() as f64 / other.0.moved_num() as f64;

                self_avg_len.partial_cmp(&other_avg_len).unwrap().reverse()
            }
        }

        impl PartialEq for CompByRevAvgLen {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl Eq for CompByRevAvgLen {}

        impl PartialOrd for CompByRevAvgLen {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CompByRevAvgLen {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
    }
}
