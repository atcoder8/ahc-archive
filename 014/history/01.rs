use std::collections::BinaryHeap;

use rect_join::{initialize_n_and_m, GridPoints};

use crate::{rect_join::CmpSumWeight, time_measurement::StopWatch};

const TIME_LIMIT: f64 = 4.5;

fn main() {
    let stop_watch = StopWatch::new();

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

    let init_grid_points = GridPoints::new(&xy);
    eprintln!("Initial Score = {}", init_grid_points.score());

    let mut best_grid_points = init_grid_points.clone();
    let mut heap = BinaryHeap::from(vec![CmpSumWeight(init_grid_points)]);

    while let Some(CmpSumWeight(curr_grid_points)) = heap.pop() {
        let selections = curr_grid_points.all_possible_selections();

        for selection in selections {
            let marked_grid_points = curr_grid_points.marked_grid_points(selection);

            if marked_grid_points.sum_weight() > best_grid_points.sum_weight() {
                best_grid_points = marked_grid_points.clone();
            }

            heap.push(CmpSumWeight(marked_grid_points));
        }

        if stop_watch.elapsed_time() > TIME_LIMIT {
            break;
        }
    }

    best_grid_points.show_history();
    eprintln!("Score = {}", best_grid_points.score());
}

pub mod time_measurement {
    //! This module provides the ability to measure execution time.

    use std::time::Instant;

    /// This structure provides the ability to measure execution time.
    pub struct StopWatch(Instant);

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
    use std::{cmp::Ordering, mem};

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

    #[derive(Debug, Hash, Clone)]
    pub struct GridLines {
        pub horizontal: FixedBitSet,
        pub vertical: FixedBitSet,
        pub upward: FixedBitSet,
        pub downward: FixedBitSet,
    }

    impl GridLines {
        pub fn new() -> Self {
            Self {
                horizontal: FixedBitSet::with_capacity(side_len() * n()),
                vertical: FixedBitSet::with_capacity(n() * side_len()),
                upward: FixedBitSet::with_capacity(side_len() * side_len()),
                downward: FixedBitSet::with_capacity(side_len() * side_len()),
            }
        }

        pub fn drawn(&self, source_coord: Coord, dir: Dir8) -> bool {
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

        pub fn draw_line(&mut self, source_coord: Coord, dir: Dir8) {
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

    #[derive(Debug, Hash, Clone)]
    pub struct GridPoints {
        marked: FixedBitSet,
        grid_lines: GridLines,
        sum_weight: usize,
        history: Vec<Selection>,
    }

    impl GridPoints {
        pub fn new(xy: &Vec<(usize, usize)>) -> Self {
            let mut marked = FixedBitSet::with_capacity(n() * n());

            for &(x, y) in xy {
                marked.insert(x * n() + y);
            }

            let mut grid_points = Self {
                marked,
                grid_lines: GridLines::new(),
                sum_weight: 0,
                history: vec![],
            };
            grid_points.sum_weight = grid_points.calc_sum_weight();

            grid_points
        }

        pub fn is_marked(&self, coord: Coord) -> bool {
            self.marked[coord.0 * n() + coord.1]
        }

        fn calc_sum_weight(&self) -> usize {
            let c = (n() - 1) / 2;

            (0..n())
                .map(|x| {
                    let diff_x = if x >= c { x - c } else { c - x };
                    let sq_diff_x = diff_x * diff_x;

                    (0..n())
                        .map(|y| {
                            if self.marked[x * n() + y] {
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
                && (y1..y2).all(|y| !self.grid_lines.drawn((x, y), Dir8::Upper))
        }

        fn horizontal_line_is_empty(&self, x_pair: (usize, usize), y: usize) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            ((x1 + 1)..x2).all(|x| !self.is_marked((x, y)))
                && (x1..x2).all(|x| !self.grid_lines.drawn((x, y), Dir8::Right))
        }

        #[allow(unused)]
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
                && (0..dist).all(|i| !self.grid_lines.drawn((sx + i, sy + i), Dir8::UpperRight))
        }

        fn downward_line_is_empty(&self, source_coord: Coord, dist: usize) -> bool {
            let (sx, sy) = source_coord;

            (1..dist).all(|i| !self.is_marked((sx + i, sy - i)))
                && (0..dist).all(|i| !self.grid_lines.drawn((sx + i, sy - i), Dir8::LowerRight))
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
                .take_while(|&x| !self.grid_lines.drawn((x, ty), Dir8::Right))
                .find(|&x| self.is_marked((x, ty)));

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = ((tx + 1)..n())
                .take_while(|&x| !self.grid_lines.drawn((x, ty), Dir8::Left))
                .find(|&x| self.is_marked((x, ty)));

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = (0..ty)
                .rev()
                .take_while(|&y| !self.grid_lines.drawn((tx, y), Dir8::Upper))
                .find(|&y| self.is_marked((tx, y)));

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = ((ty + 1)..n())
                .take_while(|&y| !self.grid_lines.drawn((tx, y), Dir8::Lower))
                .find(|&y| self.is_marked((tx, y)));

            for (other_x, other_y) in [left_x, right_x]
                .into_iter()
                .cartesian_product([upper_y, lower_y].into_iter())
            {
                if let (&Some(other_x), &Some(other_y)) = (other_x, other_y) {
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
                .take_while(|&i| !self.grid_lines.drawn((tx + i, ty + i), Dir8::LowerLeft))
                .find(|&i| self.is_marked((tx + i, ty + i)));

            // Diagonal distance of the upper left marked coordinate closest to the target coordinate.
            let upper_left_dist = (1..=tx.min(n() - 1 - ty))
                .take_while(|&i| !self.grid_lines.drawn((tx - i, ty + i), Dir8::LowerRight))
                .find(|&i| self.is_marked((tx - i, ty + i)));

            // Diagonal distance of the lower left marked coordinate closest to the target coordinate.
            let lower_left_dist = (1..=tx.min(ty))
                .take_while(|&i| !self.grid_lines.drawn((tx - i, ty - i), Dir8::UpperRight))
                .find(|&i| self.is_marked((tx - i, ty - i)));

            // Diagonal distance of the lower right marked coordinate closest to the target coordinate.
            let lower_right_dist = (1..=(n() - 1 - tx).min(ty))
                .take_while(|&i| !self.grid_lines.drawn((tx + i, ty - i), Dir8::UpperLeft))
                .find(|&dist| self.is_marked((tx + dist, ty - dist)));

            // The upper right coordinate of the marked grid point is closest to the target coordinate.
            let upper_right_coord = upper_right_dist.and_then(|dist| Some((tx + dist, ty + dist)));

            // The upper left coordinate of the marked grid point is closest to the target coordinate.
            let upper_left_coord = upper_left_dist.and_then(|dist| Some((tx - dist, ty + dist)));

            // The lower left coordinate of the marked grid point is closest to the target coordinate.
            let lower_left_coord = lower_left_dist.and_then(|dist| Some((tx - dist, ty - dist)));

            // The lower right coordinate of the marked grid point is closest to the target coordinate.
            let lower_right_coord = lower_right_dist.and_then(|dist| Some((tx + dist, ty - dist)));

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

        pub fn search_for_selection(&self, coord: Coord) -> Option<Selection> {
            if let Some(selection) = self.search_straight_selection(coord) {
                return Some(selection);
            }

            if let Some(selection) = self.search_diagonal_selection(coord) {
                return Some(selection);
            }

            None
        }

        pub fn all_possible_selections(&self) -> Vec<Selection> {
            (0..n())
                .map(|x| (0..n()).filter_map(move |y| self.search_for_selection((x, y))))
                .flatten()
                .collect()
        }

        fn draw_vertical_grid_line(&mut self, x: usize, y_pair: (usize, usize)) {
            let (mut y1, mut y2) = y_pair;
            if y1 > y2 {
                mem::swap(&mut y1, &mut y2)
            }

            for y in y1..y2 {
                self.grid_lines.draw_line((x, y), Dir8::Upper);
            }
        }

        fn draw_horizontal_grid_line(&mut self, x_pair: (usize, usize), y: usize) {
            let (mut x1, mut x2) = x_pair;
            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            for x in x1..x2 {
                self.grid_lines.draw_line((x, y), Dir8::Right);
            }
        }

        #[allow(unused)]
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
                self.grid_lines
                    .draw_line((sx + i, sy + i), Dir8::UpperRight);
            }
        }

        fn draw_downward_grid_line(&mut self, source_coord: Coord, dist: usize) {
            let (sx, sy) = source_coord;

            for i in 0..dist {
                self.grid_lines
                    .draw_line((sx + i, sy - i), Dir8::LowerRight);
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

        pub fn draw_grid_line(&mut self, coord1: Coord, coord2: Coord) {
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

        pub fn marking(&mut self, selection: Selection) {
            debug_assert!(!self.marked[selection[0].0 * n() + selection[0].1]);

            debug_assert!(selection.iter().skip(1).all(|&coord| self.is_marked(coord)));

            debug_assert!(selection.clone()
                .into_iter()
                .chain([selection[0]].into_iter())
                .tuple_windows::<(&Coord, &Coord)>()
                .all(|(&coord1, &coord2)| self.line_is_empty(coord1, coord2)));

            self.marked.insert(selection[0].0 * n() + selection[0].1);

            for (&coord1, &coord2) in selection
                .into_iter()
                .chain([selection[0]].into_iter())
                .tuple_windows::<(&Coord, &Coord)>()
            {
                self.draw_grid_line(coord1, coord2);
            }

            self.sum_weight = self.calc_sum_weight();

            self.history.push(selection);
        }

        pub fn marked_grid_points(&self, selection: Selection) -> Self {
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
    }

    #[derive(Debug, Hash, Clone)]
    pub struct CmpSumWeight(pub GridPoints);

    impl PartialEq for CmpSumWeight {
        fn eq(&self, other: &Self) -> bool {
            self.0.sum_weight == other.0.sum_weight
        }
    }

    impl Eq for CmpSumWeight {}

    impl PartialOrd for CmpSumWeight {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            if self.0.sum_weight < other.0.sum_weight {
                Some(Ordering::Less)
            } else if self.0.sum_weight > other.0.sum_weight {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Equal)
            }
        }
    }

    impl Ord for CmpSumWeight {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
}
