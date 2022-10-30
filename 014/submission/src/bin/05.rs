use std::collections::{BinaryHeap, HashSet};

use rand::seq::SliceRandom;
use rect_join::{compare::*, initialize_n_and_m, RectJoinNode, Selection};
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

    let best_grid_points = strategy3(init_grid_points);

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
    let mut heap = BinaryHeap::from(vec![CompBySumWeight(init_node)]);

    while let Some(CompBySumWeight(curr_node)) = heap.pop() {
        let selections = curr_node.all_possible_selections();

        for selection in selections {
            let marked_node = curr_node.marked_node(selection);

            if used.contains(marked_node.grid_state()) {
                continue;
            }

            used.insert(marked_node.grid_state().clone());

            if marked_node.sum_weight() > best_node.sum_weight() {
                best_node = marked_node.clone();
            }

            heap.push(CompBySumWeight(marked_node));
        }

        if stop_watch.elapsed_time() > TIME_LIMIT {
            break;
        }
    }

    loop {
        let selections = best_node.all_possible_selections();

        if selections.is_empty() {
            break;
        }

        for selection in selections {
            best_node.try_marking(selection);
        }
    }

    best_node
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
        let straight_selections = curr_node.all_possible_straight_selections();
        let diagonal_selections = curr_node.all_possible_diagonal_selections();

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
        let mut selections = best_node.all_possible_selections();

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

fn strategy3(init_node: RectJoinNode) -> RectJoinNode {
    let mut rng = rand::thread_rng();

    let stop_watch = StopWatch::new();

    let straight_selections = init_node.all_possible_straight_selections();
    let diagonal_selections = init_node.all_possible_diagonal_selections();
    let mut selections: Vec<Selection> = straight_selections
        .into_iter()
        .chain(diagonal_selections.into_iter())
        .collect();
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
        debug_assert_eq!(best_node.all_possible_selections().len(), 0);

        best_node
    } else {
        init_node
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
                }
                Dir8::Right => {
                    debug_assert!(sx < side_len() && sy < n());
                    debug_assert!(!self.horizontal[sx * n() + sy]);

                    self.horizontal.insert(sx * n() + sy);
                }
                Dir8::Lower => {
                    debug_assert!(sx < n() && 0 < sy && sy < n());
                    debug_assert!(!self.vertical[sx * side_len() + sy - 1]);

                    self.vertical.insert(sx * side_len() + sy - 1);
                }
                Dir8::Upper => {
                    debug_assert!(sx < n() && sy < side_len());
                    debug_assert!(!self.vertical[sx * side_len() + sy]);

                    self.vertical.insert(sx * side_len() + sy);
                }
                Dir8::LowerLeft => {
                    debug_assert!(0 < sx && sx < n() && 0 < sy && sy < n());
                    debug_assert!(!self.upward[(sx - 1) * side_len() + sy - 1]);

                    self.upward.insert((sx - 1) * side_len() + sy - 1);
                }
                Dir8::UpperRight => {
                    debug_assert!(sx < side_len() && sy < side_len());
                    debug_assert!(!self.upward[sx * side_len() + sy]);

                    self.upward.insert(sx * side_len() + sy);
                }
                Dir8::LowerRight => {
                    debug_assert!(sx < side_len() && 0 < sy && sy < n());
                    debug_assert!(!self.downward[sx * side_len() + sy - 1]);

                    self.downward.insert(sx * side_len() + sy - 1);
                }
                Dir8::UpperLeft => {
                    debug_assert!(0 < sx && sx < n() && sy < side_len());
                    debug_assert!(!self.downward[(sx - 1) * side_len() + sy]);

                    self.downward.insert((sx - 1) * side_len() + sy);
                }
            }
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct GridState {
        pub marks: FixedBitSet,
        pub lines: GridLines,
        pub sum_weight: usize,
    }

    impl GridState {
        fn new(init_marked_coords: &Vec<(usize, usize)>) -> Self {
            let mut marks = FixedBitSet::with_capacity(n() * n());

            for &(x, y) in init_marked_coords {
                marks.insert(x * n() + y);
            }

            let sum_weight = calc_sum_weight(&marks);

            Self {
                marks,
                lines: GridLines::new(),
                sum_weight,
            }
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

        fn vertical_line_is_empty(&self, x: usize, y_pair: (usize, usize)) -> bool {
            let (mut y1, mut y2) = y_pair;

            debug_assert_ne!(y1, y2);

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            ((y1 + 1)..y2).all(|y| !self.is_marked((x, y)))
                && (y1..y2).all(|y| !self.lines.is_drawn((x, y), Dir8::Upper))
        }

        fn horizontal_line_is_empty(&self, x_pair: (usize, usize), y: usize) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            ((x1 + 1)..x2).all(|x| !self.is_marked((x, y)))
                && (x1..x2).all(|x| !self.lines.is_drawn((x, y), Dir8::Right))
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

            (1..dist).all(|i| !self.is_marked((sx + i, sy + i)))
                && (0..dist).all(|i| !self.lines.is_drawn((sx + i, sy + i), Dir8::UpperRight))
        }

        fn downward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            (1..dist).all(|i| !self.is_marked((sx + i, sy - i)))
                && (0..dist).all(|i| !self.lines.is_drawn((sx + i, sy - i), Dir8::LowerRight))
        }

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

        fn search_straight_selection(&self, coord: Coord) -> Option<Selection> {
            if self.is_marked(coord) {
                return None;
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            // x-coordinate of the marked left closest to the target coordinate.
            let left_x = (0..tx)
                .rev()
                .take_while(|&x| !self.lines.is_drawn((x, ty), Dir8::Right))
                .find(|&x| self.is_marked((x, ty)));

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = ((tx + 1)..n())
                .take_while(|&x| !self.lines.is_drawn((x, ty), Dir8::Left))
                .find(|&x| self.is_marked((x, ty)));

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = (0..ty)
                .rev()
                .take_while(|&y| !self.lines.is_drawn((tx, y), Dir8::Upper))
                .find(|&y| self.is_marked((tx, y)));

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = ((ty + 1)..n())
                .take_while(|&y| !self.lines.is_drawn((tx, y), Dir8::Lower))
                .find(|&y| self.is_marked((tx, y)));

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

        fn search_diagonal_selection(&self, coord: Coord) -> Option<Selection> {
            if self.is_marked(coord) {
                return None;
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord;

            // Diagonal distance of the upper right marked coordinate closest to the target coordinate.
            let upper_right_dist = (1..=(n() - 1 - tx).min(n() - 1 - ty))
                .take_while(|&i| !self.lines.is_drawn((tx + i, ty + i), Dir8::LowerLeft))
                .find(|&i| self.is_marked((tx + i, ty + i)));

            // Diagonal distance of the upper left marked coordinate closest to the target coordinate.
            let upper_left_dist = (1..=tx.min(n() - 1 - ty))
                .take_while(|&i| !self.lines.is_drawn((tx - i, ty + i), Dir8::LowerRight))
                .find(|&i| self.is_marked((tx - i, ty + i)));

            // Diagonal distance of the lower left marked coordinate closest to the target coordinate.
            let lower_left_dist = (1..=tx.min(ty))
                .take_while(|&i| !self.lines.is_drawn((tx - i, ty - i), Dir8::UpperRight))
                .find(|&i| self.is_marked((tx - i, ty - i)));

            // Diagonal distance of the lower right marked coordinate closest to the target coordinate.
            let lower_right_dist = (1..=(n() - 1 - tx).min(ty))
                .take_while(|&i| !self.lines.is_drawn((tx + i, ty - i), Dir8::UpperLeft))
                .find(|&dist| self.is_marked((tx + dist, ty - dist)));

            // The upper right coordinate of the marked grid point is closest to the target coordinate.
            let upper_right_coord = upper_right_dist.map(|dist| (tx + dist, ty + dist));

            // The upper left coordinate of the marked grid point is closest to the target coordinate.
            let upper_left_coord = upper_left_dist.map(|dist| (tx - dist, ty + dist));

            // The lower left coordinate of the marked grid point is closest to the target coordinate.
            let lower_left_coord = lower_left_dist.map(|dist| (tx - dist, ty - dist));

            // The lower right coordinate of the marked grid point is closest to the target coordinate.
            let lower_right_coord = lower_right_dist.map(|dist| (tx + dist, ty - dist));

            let diagonal_coords = [
                upper_right_coord,
                upper_left_coord,
                lower_left_coord,
                lower_right_coord,
                upper_right_coord,
            ];

            for window in diagonal_coords.windows(2) {
                if let (Some(coord1), Some(coord2)) = (window[0], window[1]) {
                    let (x1, y1) = coord1;
                    let (x2, y2) = coord2;

                    let opposite_x = if let Some(opposite_x) = (x1 + x2).checked_sub(tx) {
                        if opposite_x < n() {
                            opposite_x
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    let opposite_y = if let Some(opposite_y) = (y1 + y2).checked_sub(ty) {
                        if opposite_y < n() {
                            opposite_y
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    let opposite_coord = (opposite_x, opposite_y);

                    if self.is_marked(opposite_coord)
                        && self.diagonal_line_is_empty(coord1, opposite_coord)
                        && self.diagonal_line_is_empty(coord2, opposite_coord)
                    {
                        return Some([coord, coord1, opposite_coord, coord2]);
                    }
                }
            }

            // Returns `None` if the target coordinates cannot be marked.
            None
        }

        fn search_selection(&self, coord: Coord) -> Option<Selection> {
            if let Some(selection) = self.search_straight_selection(coord) {
                return Some(selection);
            }

            if let Some(selection) = self.search_diagonal_selection(coord) {
                return Some(selection);
            }

            None
        }

        fn all_possible_straight_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| (0..n()).filter_map(move |y| self.search_straight_selection((x, y))))
                .collect()
        }

        fn all_possible_diagonal_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| (0..n()).filter_map(move |y| self.search_diagonal_selection((x, y))))
                .collect()
        }

        fn all_possible_selections(&self) -> Vec<Selection> {
            (0..n())
                .flat_map(|x| (0..n()).filter_map(move |y| self.search_selection((x, y))))
                .collect()
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
            !self.marks[selection[0].0 * n() + selection[0].1]
                && selection.iter().skip(1).all(|&coord| self.is_marked(coord))
                && (0..selection.len())
                    .all(|i| self.line_is_empty(selection[i], selection[(i + 1) % selection.len()]))
        }

        fn marking(&mut self, selection: Selection) {
            debug_assert!(self.selectable(selection));

            self.marks.insert(selection[0].0 * n() + selection[0].1);

            for i in 0..selection.len() {
                self.draw_grid_line(selection[i], selection[(i + 1) % selection.len()]);
            }

            self.sum_weight += weight(selection[0]);
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct RectJoinNode {
        grid_state: GridState,
        history: Vec<Selection>,
    }

    impl RectJoinNode {
        pub fn new(xy: &Vec<(usize, usize)>) -> Self {
            let mut marked = FixedBitSet::with_capacity(n() * n());

            for &(x, y) in xy {
                marked.insert(x * n() + y);
            }

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
    }
}

