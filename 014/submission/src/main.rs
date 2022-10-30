use crate::{
    mountain_climbing::mountain_climbing,
    rect_join::{
        cmp_node::{CmpByRevDrawnLen, CmpBySumWeight},
        coordinate::CoordElemByU8,
        *,
    },
    time_measurement::StopWatch,
};

const TIME_LIMIT: f64 = 4.5;

fn main() {
    let stop_watch = StopWatch::new();

    let (n, m) = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        (
            iter.next().unwrap().parse::<CoordElemByU8>().unwrap(),
            iter.next().unwrap().parse::<usize>().unwrap(),
        )
    };

    initialize_n_and_m(n, m);

    let mut xy = Vec::new();
    for _ in 0..m {
        xy.push({
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            let mut iter = line.split_whitespace();
            (
                iter.next().unwrap().parse::<CoordElemByU8>().unwrap(),
                iter.next().unwrap().parse::<CoordElemByU8>().unwrap(),
            )
                .into()
        });
    }

    let init_node = RectJoinNode::new(&xy);

    set_initial_weight(init_node.grid_state().marks().sum_weight());

    eprintln!("Initial Score = {}", initial_score());

    let mut best_nodes = vec![];

    let best_node = mountain_climbing::<_, CmpBySumWeight>(init_node.clone(), |node| {
        node.grid_state().all_drawable_rectangles()
    })
    .1;
    best_nodes.push(best_node);

    let best_node = mountain_climbing::<_, CmpByRevDrawnLen>(init_node.clone(), |node| {
        node.grid_state().all_drawable_rectangles()
    })
    .1;
    best_nodes.push(best_node);

    let best_node = {
        let mut cur_node = init_node.clone();

        while {
            let (updated_flag_1, next_node) =
                mountain_climbing::<_, CmpByRevDrawnLen>(cur_node, |node| {
                    node.grid_state().all_drawable_straight_rectangles()
                });

            let (updated_flag_2, next_node) =
                mountain_climbing::<_, CmpByRevDrawnLen>(next_node, |node| {
                    node.grid_state().all_drawable_diagonal_rectangles()
                });

            cur_node = next_node;

            updated_flag_1 || updated_flag_2
        } {}

        cur_node
    };
    best_nodes.push(best_node);

    for beam_width in 1.. {
        if stop_watch.elapsed_time() > TIME_LIMIT {
            break;
        }

        let best_node = {
            let best_node = beam::beam_search::<CmpByRevDrawnLen>(
                vec![init_node.clone()],
                beam_width,
                TIME_LIMIT,
                &stop_watch,
            )
            .0;

            let best_node = if let Some(best_node) = best_node {
                best_node
            } else {
                continue;
            };

            mountain_climbing::<_, CmpByRevDrawnLen>(best_node, |node| {
                node.grid_state().all_drawable_rectangles()
            })
            .1
        };

        best_nodes.push(best_node);
    }

    let best_node = best_nodes
        .into_iter()
        .max_by_key(|node| node.sum_weight())
        .unwrap();

    best_node.show_history();

    eprintln!("Number of moves = {}", best_node.moved_num());
    eprintln!("Score = {}", best_node.score());
}

pub mod mountain_climbing {
    use crate::rect_join::{cmp_node::CmpNode, Rect, RectJoinNode};

    pub fn mountain_climbing<GetRects, CmpBy>(
        init_node: RectJoinNode,
        get_rects: GetRects,
    ) -> (bool, RectJoinNode)
    where
        GetRects: FnMut(&RectJoinNode) -> Vec<Rect> + Clone,
        CmpBy: CmpNode,
    {
        let mut cur_node = init_node;
        let mut updated_flag = false;

        loop {
            let rects = get_rects.clone()(&cur_node);

            if rects.is_empty() {
                break;
            }

            cur_node = rects
                .into_iter()
                .map(|rect| CmpBy::from(cur_node.marked_node(rect)))
                .max()
                .unwrap()
                .into_node();

            updated_flag = true;
        }

        (updated_flag, cur_node)
    }
}

pub mod beam {
    use std::collections::BinaryHeap;

    use crate::{
        rect_join::{cmp_node::*, *},
        time_measurement::StopWatch,
    };

    pub fn beam_search<CmpBy>(
        nodes: Vec<RectJoinNode>,
        beam_width: usize,
        time_limit: f64,
        stop_watch: &StopWatch,
    ) -> (Option<RectJoinNode>, Vec<RectJoinNode>)
    where
        CmpBy: CmpNode,
    {
        let mut best_node: Option<RectJoinNode> = None;

        let mut beam = BinaryHeap::new();
        beam.extend(nodes.into_iter().map(|node| CmpBy::from(node)));

        while !beam.is_empty() && stop_watch.elapsed_time() <= time_limit {
            let mut next_beam = BinaryHeap::new();

            let iter = beam.into_iter().filter_map(|cmp_by| {
                let node = cmp_by.into_node();
                let rects = node.grid_state().all_drawable_rectangles();

                if rects.is_empty() {
                    None
                } else {
                    Some((node, rects))
                }
            });

            for (node, rects) in iter.take(beam_width) {
                if stop_watch.elapsed_time() > time_limit {
                    break;
                }

                next_beam.extend(
                    rects
                        .into_iter()
                        .map(|rect| CmpBy::from(node.marked_node(rect))),
                );

                if let Some(best_node) = &mut best_node {
                    if node.sum_weight() > best_node.sum_weight() {
                        *best_node = node;
                    }
                } else {
                    best_node = Some(node)
                }
            }

            beam = next_beam
        }

        (
            best_node,
            beam.into_iter().map(|cmp_by| cmp_by.into_node()).collect(),
        )
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

    use self::coordinate::{CoordByU16, CoordElemByU8};

    pub type WeightByU32 = u32;

    pub type ScoreByU32 = u32;

    pub type DrawnDistByU16 = u16;

    /// Number of vertical and horizontal grid points of a square grid
    pub static mut N: Option<CoordElemByU8> = None;

    /// Number of grid points marked initially
    pub static mut M: Option<usize> = None;

    static mut SCORE_UNIT: Option<f64> = None;

    pub static mut INITIAL_WEIGHT: Option<WeightByU32> = None;

    pub static mut INITIAL_SCORE: Option<ScoreByU32> = None;

    pub fn initialize_n_and_m(n: CoordElemByU8, m: usize) {
        unsafe {
            N = Some(n);
            M = Some(m);
            SCORE_UNIT = Some(6e6 / (m as f64 * ((n as f64).powi(2) + 5.0) as f64))
        }
    }

    pub fn n() -> CoordElemByU8 {
        unsafe { N.unwrap() }
    }

    pub fn side_len() -> CoordElemByU8 {
        n() - 1
    }

    pub fn m() -> usize {
        unsafe { M.unwrap() }
    }

    fn score_unit() -> f64 {
        unsafe { SCORE_UNIT.unwrap() }
    }

    pub fn max_sum_weight() -> WeightByU32 {
        let sq_n = (n() as WeightByU32).pow(2);

        sq_n * (sq_n + 5) / 6
    }

    pub fn max_score() -> ScoreByU32 {
        (score_unit() * max_sum_weight() as f64).round() as ScoreByU32
    }

    pub fn set_initial_weight(init_weight: WeightByU32) {
        unsafe {
            INITIAL_WEIGHT = Some(init_weight);
            INITIAL_SCORE = Some(((init_weight as f64) * score_unit()).round() as ScoreByU32);
        }
    }

    pub fn initial_weight() -> WeightByU32 {
        unsafe { INITIAL_WEIGHT.unwrap() }
    }

    pub fn initial_score() -> ScoreByU32 {
        unsafe { INITIAL_SCORE.unwrap() }
    }

    pub mod coordinate {
        use super::{n, WeightByU32};

        pub type CoordElemByU8 = u8;

        pub type CoordByU8Pair = (CoordElemByU8, CoordElemByU8);

        pub type CoordByUsizePair = (usize, usize);

        #[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
        pub struct CoordByU16(pub u16);

        impl CoordByU16 {
            const SHIFT: u8 = 8;
            const MASK: u16 = 255;

            pub fn center() -> Self {
                CoordByU16::from_coord_idx((n() as usize).pow(2) / 2)
            }

            pub fn from_coord_idx(coord_idx: usize) -> Self {
                debug_assert!(coord_idx < (n() as usize).pow(2));

                let coord = (
                    (coord_idx / n() as usize) as CoordElemByU8,
                    (coord_idx % n() as usize) as CoordElemByU8,
                );

                CoordByU16::from(coord)
            }

            pub fn divide_u8(self) -> (u8, u8) {
                (
                    (self.0 >> CoordByU16::SHIFT) as u8,
                    (self.0 & CoordByU16::MASK) as u8,
                )
            }

            pub fn divide_usize(self) -> (usize, usize) {
                (
                    (self.0 >> CoordByU16::SHIFT) as usize,
                    (self.0 & CoordByU16::MASK) as usize,
                )
            }

            pub fn chebyshev_distance(self, other: Self) -> u8 {
                let (x1, y1) = self.divide_u8();
                let (x2, y2) = other.divide_u8();

                let diff_x = if x1 >= x2 { x1 - x2 } else { x2 - x1 };
                let diff_y = if y1 >= y2 { y1 - y2 } else { y2 - y1 };

                diff_x.max(diff_y)
            }

            pub fn manhattan_distance(self, other: Self) -> u8 {
                let (x1, y1) = self.divide_u8();
                let (x2, y2) = other.divide_u8();

                let diff_x = if x1 >= x2 { x1 - x2 } else { x2 - x1 };
                let diff_y = if y1 >= y2 { y1 - y2 } else { y2 - y1 };

                diff_x + diff_y
            }

            pub fn square_euclid_distance(self, other: Self) -> u16 {
                let (x1, y1) = self.divide_u8();
                let (x2, y2) = other.divide_u8();

                let diff_x = if x1 >= x2 { x1 - x2 } else { x2 - x1 };
                let diff_y = if y1 >= y2 { y1 - y2 } else { y2 - y1 };

                (diff_x as u16).pow(2) + (diff_y as u16).pow(2)
            }

            pub fn euclid_distance(self, other: Self) -> f64 {
                (self.square_euclid_distance(other) as f64).sqrt()
            }

            pub fn weight(self) -> WeightByU32 {
                let c = n() / 2;
                let (x, y) = self.divide_u8();

                let diff_x = if x >= c { x - c } else { c - x };
                let sq_diff_x = (diff_x as WeightByU32).pow(2);

                let diff_y = if y >= c { y - c } else { c - y };
                let sq_diff_y = (diff_y as WeightByU32).pow(2);

                sq_diff_x + sq_diff_y + 1
            }

            pub fn to_coord_idx(self) -> usize {
                let (x, y) = self.divide_usize();

                x * n() as usize + y
            }
        }

        impl From<CoordByU8Pair> for CoordByU16 {
            fn from(coord: CoordByU8Pair) -> Self {
                let (x, y) = coord;

                debug_assert!(x < n() && y < n());

                Self(((x as u16) << CoordByU16::SHIFT) | y as u16)
            }
        }

        impl From<CoordByUsizePair> for CoordByU16 {
            fn from(coord: CoordByUsizePair) -> Self {
                let (x, y) = coord;

                debug_assert!((x as CoordElemByU8) < n() && (y as CoordElemByU8) < n());

                Self(((x as u16) << CoordByU16::SHIFT) | y as u16)
            }
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct RectCorners {
        target: CoordByU16,
        adjacent1: CoordByU16,
        opposite: CoordByU16,
        adjacent2: CoordByU16,
    }

    impl RectCorners {
        pub fn new(
            target: CoordByU16,
            adjacent1: CoordByU16,
            opposite: CoordByU16,
            adjacent2: CoordByU16,
        ) -> Self {
            Self {
                target,
                adjacent1,
                opposite,
                adjacent2,
            }
        }

        pub fn show(&self) {
            let (tar_x, tar_y) = self.target.divide_u8();
            let (adj1_x, adj1_y) = self.adjacent1.divide_u8();
            let (opp_x, opp_y) = self.opposite.divide_u8();
            let (adj2_x, adj2_y) = self.adjacent2.divide_u8();

            println!(
                "{} {} {} {} {} {} {} {}",
                tar_x, tar_y, adj1_x, adj1_y, opp_x, opp_y, adj2_x, adj2_y
            );
        }

        pub fn target(&self) -> CoordByU16 {
            self.target
        }

        pub fn draw_distance(&self) -> DrawnDistByU16 {
            self.target.chebyshev_distance(self.opposite) as DrawnDistByU16
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub enum Rect {
        Straight(RectCorners),
        Diagonal(RectCorners),
    }

    impl Rect {
        pub fn corners(&self) -> &RectCorners {
            match self {
                Rect::Straight(corners) => corners,
                Rect::Diagonal(corners) => corners,
            }
        }

        pub fn show(&self) {
            self.corners().show()
        }

        pub fn target(&self) -> CoordByU16 {
            self.corners().target()
        }

        pub fn draw_dist(&self) -> DrawnDistByU16 {
            self.corners().draw_distance()
        }

        pub fn corner_coords(&self) -> [CoordByU16; 4] {
            let corners = self.corners();

            [
                corners.target,
                corners.adjacent1,
                corners.opposite,
                corners.adjacent2,
            ]
        }

        pub fn is_straight(&self) -> bool {
            if let Rect::Straight(_) = self {
                true
            } else {
                false
            }
        }

        pub fn is_diagonal(&self) -> bool {
            if let Rect::Diagonal(_) = self {
                true
            } else {
                false
            }
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct Marks {
        pub marks: FixedBitSet,
        pub x_range: (CoordElemByU8, CoordElemByU8),
        pub y_range: (CoordElemByU8, CoordElemByU8),
        pub horizontal_existences: u64,
        pub vertical_existences: u64,
        pub upward_existences: u128,
        pub downward_existences: u128,
        pub sum_weight: WeightByU32,
    }

    impl Marks {
        fn add_mark(&mut self, coord: CoordByU16) {
            debug_assert!(!self.is_marked(coord));

            self.marks.insert(coord.to_coord_idx());

            let (x, y) = coord.divide_u8();

            self.x_range.0 = self.x_range.0.min(x);
            self.x_range.1 = self.x_range.1.max(x);

            self.y_range.0 = self.y_range.0.min(y);
            self.y_range.1 = self.y_range.1.max(y);

            let (x, y) = (x as usize, y as usize);

            self.vertical_existences |= 1 << x;
            self.horizontal_existences |= 1 << y;
            self.upward_existences |= 1 << (n() as usize - 1 + x - y);
            self.downward_existences |= 1 << (x + y);

            // self.vertical_existences[x] += 1;
            // self.horizontal_existences[y] += 1;
            // self.upward_existences[n() as usize - 1 + x - y] += 1;
            // self.downward_existences[x + y] += 1;

            self.sum_weight += coord.weight();
        }

        fn new(init_marked_coords: &Vec<CoordByU16>) -> Self {
            let n = n() as usize;

            let (front_x, front_y) = init_marked_coords[0].divide_u8();

            let mut marks = Self {
                marks: FixedBitSet::with_capacity(n.pow(2)),
                x_range: (front_x, front_x),
                y_range: (front_y, front_y),
                horizontal_existences: 0,
                vertical_existences: 0,
                upward_existences: 0,
                downward_existences: 0,
                sum_weight: 0,
            };

            for &coord in init_marked_coords {
                marks.add_mark(coord);
            }

            marks
        }

        pub fn is_marked(&self, coord: CoordByU16) -> bool {
            self.marks[coord.to_coord_idx()]
        }

        pub fn sum_weight(&self) -> u32 {
            self.sum_weight
        }

        pub fn score(&self) -> ScoreByU32 {
            (score_unit() * self.sum_weight() as f64).round() as ScoreByU32
        }

        fn all_vertical_line_is_empty(&self, x: CoordElemByU8) -> bool {
            debug_assert!(x < n());

            (self.vertical_existences >> x) & 1 == 0
            // self.vertical_existences[x as usize] == 0
        }

        fn all_horizontal_line_is_empty(&self, y: CoordElemByU8) -> bool {
            debug_assert!(y < n());

            (self.horizontal_existences >> y) & 1 == 0
            // self.horizontal_existences[y as usize] == 0
        }

        fn all_upward_line_is_empty(&self, coord: CoordByU16) -> bool {
            let (x, y) = coord.divide_u8();

            debug_assert!(x < n() && y < n());

            (self.upward_existences >> (n() - 1 + x - y)) & 1 == 0
            // self.upward_existences[(n() - 1 + x - y) as usize] == 0
        }

        fn all_downward_line_is_empty(&self, coord: CoordByU16) -> bool {
            let (x, y) = coord.divide_u8();

            (self.downward_existences >> (x + y)) & 1 == 0
            // self.downward_existences[(x + y) as usize] == 0
        }

        fn vertical_line_is_empty(
            &self,
            x: CoordElemByU8,
            y_pair: (CoordElemByU8, CoordElemByU8),
        ) -> bool {
            let (mut y1, mut y2) = y_pair;

            debug_assert_ne!(y1, y2);

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            self.all_vertical_line_is_empty(x)
                || ((y1 + 1)..y2).all(|y| !self.is_marked((x, y).into()))
        }

        fn horizontal_line_is_empty(
            &self,
            x_pair: (CoordElemByU8, CoordElemByU8),
            y: CoordElemByU8,
        ) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            self.all_horizontal_line_is_empty(y)
                || ((x1 + 1)..x2).all(|x| !self.is_marked((x, y).into()))
        }

        #[allow(dead_code)]
        fn straight_line_is_empty(&self, coord1: CoordByU16, coord2: CoordByU16) -> bool {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        fn upward_line_is_empty(&self, source_coord: CoordByU16, dist: CoordElemByU8) -> bool {
            let (sx, sy) = source_coord.divide_u8();

            self.all_upward_line_is_empty(source_coord)
                || (1..dist).all(|i| !self.is_marked((sx + i, sy + i).into()))
        }

        fn downward_line_is_empty(&self, source_coord: CoordByU16, dist: CoordElemByU8) -> bool {
            let (sx, sy) = source_coord.divide_u8();

            self.all_downward_line_is_empty(source_coord)
                || (1..dist).all(|i| !self.is_marked((sx + i, sy - i).into()))
        }

        #[allow(dead_code)]
        fn diagonal_line_is_empty(&self, mut coord1: CoordByU16, mut coord2: CoordByU16) -> bool {
            debug_assert_ne!(coord1, coord2);

            if coord1.0 > coord2.0 {
                mem::swap(&mut coord1, &mut coord2)
            }

            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

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
        fn line_is_empty(&self, coord1: CoordByU16, coord2: CoordByU16) -> bool {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                self.diagonal_line_is_empty(coord1, coord2)
            }
        }

        // pub fn vertical_marked_num(&self, x: CoordElemByU8) -> u8 {
        //     self.vertical_existences[x as usize]
        // }

        // pub fn horizontal_marked_num(&self, y: CoordElemByU8) -> u8 {
        //     self.horizontal_existences[y as usize]
        // }

        // pub fn upward_marked_num(&self, coord: CoordByU16) -> u8 {
        //     let (x, y) = coord.divide_usize();

        //     self.upward_existences[n() as usize - 1 + x - y]
        // }

        // pub fn downward_marked_num(&self, coord: CoordByU16) -> u8 {
        //     let (x, y) = coord.divide_usize();

        //     self.downward_existences[x + y]
        // }

        pub fn x_range(&self) -> (CoordElemByU8, CoordElemByU8) {
            self.x_range
        }

        pub fn y_range(&self) -> (CoordElemByU8, CoordElemByU8) {
            self.y_range
        }

        pub fn x_interval(&self) -> u8 {
            self.x_range.1 - self.x_range.0
        }

        pub fn y_interval(&self) -> u8 {
            self.y_range.1 - self.y_range.0
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
        pub straight_drawn_len: DrawnDistByU16,
        pub diagonal_drawn_len: DrawnDistByU16,
    }

    impl Default for GridLines {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GridLines {
        fn new() -> Self {
            let straight_capacity = (side_len() as usize) * (n() as usize);
            let diagonal_capacity = (side_len() as usize).pow(2);

            Self {
                horizontal: FixedBitSet::with_capacity(straight_capacity),
                vertical: FixedBitSet::with_capacity(straight_capacity),
                upward: FixedBitSet::with_capacity(diagonal_capacity),
                downward: FixedBitSet::with_capacity(diagonal_capacity),
                straight_drawn_len: 0,
                diagonal_drawn_len: 0,
            }
        }

        fn is_drawn(&self, source_coord: CoordByU16, dir: Dir8) -> bool {
            let n = n() as usize;
            let side_len = n - 1;

            let (sx, sy) = source_coord.divide_usize();

            match dir {
                Dir8::Left => {
                    debug_assert!(0 < sx && sx < n && sy < n);

                    self.horizontal[(sx - 1) * n + sy]
                }
                Dir8::Right => {
                    debug_assert!(sx < side_len && sy < n);

                    self.horizontal[sx * n + sy]
                }
                Dir8::Lower => {
                    debug_assert!(sx < n && 0 < sy && sy < n);

                    self.vertical[sx * side_len + sy - 1]
                }
                Dir8::Upper => {
                    debug_assert!(sx < n && sy < side_len);

                    self.vertical[sx * side_len + sy]
                }
                Dir8::LowerLeft => {
                    debug_assert!(0 < sx && sx < n && 0 < sy && sy < n);

                    self.upward[(sx - 1) * side_len + sy - 1]
                }
                Dir8::UpperRight => {
                    debug_assert!(sx < side_len && sy < side_len);

                    self.upward[sx * side_len + sy]
                }
                Dir8::LowerRight => {
                    debug_assert!(sx < side_len && 0 < sy && sy < n);

                    self.downward[sx * side_len + sy - 1]
                }
                Dir8::UpperLeft => {
                    debug_assert!(0 < sx && sx < n && sy < side_len);

                    self.downward[(sx - 1) * side_len + sy]
                }
            }
        }

        fn draw_line(&mut self, source_coord: CoordByU16, dir: Dir8) {
            let n = n() as usize;
            let side_len = n - 1;

            let (sx, sy) = source_coord.divide_usize();

            match dir {
                Dir8::Left => {
                    debug_assert!(0 < sx && sx < n && sy < n);
                    debug_assert!(!self.horizontal[(sx - 1) * n + sy]);

                    self.horizontal.insert((sx - 1) * n + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::Right => {
                    debug_assert!(sx < side_len && sy < n);
                    debug_assert!(!self.horizontal[sx * n + sy]);

                    self.horizontal.insert(sx * n + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::Lower => {
                    debug_assert!(sx < n && 0 < sy && sy < n);
                    debug_assert!(!self.vertical[sx * side_len + sy - 1]);

                    self.vertical.insert(sx * side_len + sy - 1);
                    self.straight_drawn_len += 1;
                }
                Dir8::Upper => {
                    debug_assert!(sx < n && sy < side_len);
                    debug_assert!(!self.vertical[sx * side_len + sy]);

                    self.vertical.insert(sx * side_len + sy);
                    self.straight_drawn_len += 1;
                }
                Dir8::LowerLeft => {
                    debug_assert!(0 < sx && sx < n && 0 < sy && sy < n);
                    debug_assert!(!self.upward[(sx - 1) * side_len + sy - 1]);

                    self.upward.insert((sx - 1) * side_len + sy - 1);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::UpperRight => {
                    debug_assert!(sx < side_len && sy < side_len);
                    debug_assert!(!self.upward[sx * side_len + sy]);

                    self.upward.insert(sx * side_len + sy);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::LowerRight => {
                    debug_assert!(sx < side_len && 0 < sy && sy < n);
                    debug_assert!(!self.downward[sx * side_len + sy - 1]);

                    self.downward.insert(sx * side_len + sy - 1);
                    self.diagonal_drawn_len += 1;
                }
                Dir8::UpperLeft => {
                    debug_assert!(0 < sx && sx < n && sy < side_len);
                    debug_assert!(!self.downward[(sx - 1) * side_len + sy]);

                    self.downward.insert((sx - 1) * side_len + sy);
                    self.diagonal_drawn_len += 1;
                }
            }
        }

        fn vertical_line_is_empty(
            &self,
            x: CoordElemByU8,
            y_pair: (CoordElemByU8, CoordElemByU8),
        ) -> bool {
            let (mut y1, mut y2) = y_pair;

            debug_assert_ne!(y1, y2);

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            (y1..y2).all(|y| !self.is_drawn((x, y).into(), Dir8::Upper))
        }

        fn horizontal_line_is_empty(
            &self,
            x_pair: (CoordElemByU8, CoordElemByU8),
            y: CoordElemByU8,
        ) -> bool {
            let (mut x1, mut x2) = x_pair;

            debug_assert_ne!(x1, x2);

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            (x1..x2).all(|x| !self.is_drawn((x, y).into(), Dir8::Right))
        }

        fn upward_line_is_empty(&self, source_coord: CoordByU16, dist: CoordElemByU8) -> bool {
            let (sx, sy) = source_coord.divide_u8();

            (0..dist).all(|i| !self.is_drawn((sx + i, sy + i).into(), Dir8::UpperRight))
        }

        fn downward_line_is_empty(&self, source_coord: CoordByU16, dist: CoordElemByU8) -> bool {
            let (sx, sy) = source_coord.divide_u8();

            (0..dist).all(|i| !self.is_drawn((sx + i, sy - i).into(), Dir8::LowerRight))
        }

        pub fn straight_drawn_len(&self) -> DrawnDistByU16 {
            self.straight_drawn_len
        }

        pub fn diagonal_drawn_len(&self) -> DrawnDistByU16 {
            self.diagonal_drawn_len
        }

        pub fn drawn_len(&self) -> DrawnDistByU16 {
            self.straight_drawn_len + self.diagonal_drawn_len
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct GridState {
        pub marks: Marks,
        pub lines: GridLines,
        pub markable_num: Option<usize>,
    }

    impl GridState {
        fn new(init_marked_coords: &Vec<CoordByU16>) -> Self {
            Self {
                marks: Marks::new(init_marked_coords),
                lines: GridLines::new(),
                markable_num: None,
            }
        }

        pub fn marks(&self) -> &Marks {
            &self.marks
        }

        pub fn lines(&self) -> &GridLines {
            &self.lines
        }

        pub fn vertical_line_is_empty(
            &self,
            x: CoordElemByU8,
            y_pair: (CoordElemByU8, CoordElemByU8),
        ) -> bool {
            let (mut y1, mut y2) = y_pair;

            if y1 > y2 {
                mem::swap(&mut y1, &mut y2);
            }

            self.marks.vertical_line_is_empty(x, y_pair)
                && self.lines.vertical_line_is_empty(x, y_pair)
        }

        pub fn horizontal_line_is_empty(
            &self,
            x_pair: (CoordElemByU8, CoordElemByU8),
            y: CoordElemByU8,
        ) -> bool {
            let (mut x1, mut x2) = x_pair;

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            self.marks.horizontal_line_is_empty(x_pair, y)
                && self.lines.horizontal_line_is_empty(x_pair, y)
        }

        pub fn straight_line_is_empty(&self, coord1: CoordByU16, coord2: CoordByU16) -> bool {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        pub fn upward_line_is_empty(&self, source_coord: CoordByU16, dist: CoordElemByU8) -> bool {
            self.marks.upward_line_is_empty(source_coord, dist)
                && self.lines.upward_line_is_empty(source_coord, dist)
        }

        pub fn downward_line_is_empty(
            &self,
            source_coord: CoordByU16,
            dist: CoordElemByU8,
        ) -> bool {
            self.marks.downward_line_is_empty(source_coord, dist)
                && self.lines.downward_line_is_empty(source_coord, dist)
        }

        pub fn diagonal_line_is_empty(
            &self,
            mut coord1: CoordByU16,
            mut coord2: CoordByU16,
        ) -> bool {
            debug_assert_ne!(coord1, coord2);

            if coord1.0 > coord2.0 {
                mem::swap(&mut coord1, &mut coord2)
            }

            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            let dist = x2 - x1;

            if y1 < y2 {
                debug_assert_eq!(y2 - y1, dist);

                self.upward_line_is_empty(coord1, dist)
            } else {
                debug_assert_eq!(y1 - y2, dist);

                self.downward_line_is_empty(coord1, dist)
            }
        }

        pub fn line_is_empty(&self, coord1: CoordByU16, coord2: CoordByU16) -> bool {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.vertical_line_is_empty(x1, (y1, y2))
            } else if y1 == y2 {
                self.horizontal_line_is_empty((x1, x2), y1)
            } else {
                self.diagonal_line_is_empty(coord1, coord2)
            }
        }

        // Searches for the marked left x-coordinate closest to the target coordinate.
        pub fn search_closest_left(&self, coord: CoordByU16) -> Option<CoordElemByU8> {
            let (tx, ty) = coord.divide_u8();

            if self.marks.all_horizontal_line_is_empty(ty) {
                return None;
            }

            (0..tx)
                .rev()
                .take_while(|&x| !self.lines.is_drawn((x, ty).into(), Dir8::Right))
                .find(|&x| self.marks().is_marked((x, ty).into()))
        }

        // Searches for the marked right x-coordinate closest to the target coordinate.
        pub fn search_closest_right(&self, coord: CoordByU16) -> Option<CoordElemByU8> {
            let (tx, ty) = coord.divide_u8();

            if self.marks.all_horizontal_line_is_empty(ty) {
                return None;
            }

            ((tx + 1)..n())
                .take_while(|&x| !self.lines.is_drawn((x, ty).into(), Dir8::Left))
                .find(|&x| self.marks().is_marked((x, ty).into()))
        }

        // Searches for the marked lower y-coordinate closest to the target coordinate.
        pub fn search_closest_lower(&self, coord: CoordByU16) -> Option<CoordElemByU8> {
            let (tx, ty) = coord.divide_u8();

            if self.marks.all_vertical_line_is_empty(tx) {
                return None;
            }

            (0..ty)
                .rev()
                .take_while(|&y| !self.lines.is_drawn((tx, y).into(), Dir8::Upper))
                .find(|&y| self.marks().is_marked((tx, y).into()))
        }

        // Searches for the marked upper y-coordinate closest to the target coordinate.
        pub fn search_closest_upper(&self, coord: CoordByU16) -> Option<CoordElemByU8> {
            let (tx, ty) = coord.divide_u8();

            if self.marks.all_vertical_line_is_empty(tx) {
                return None;
            }

            ((ty + 1)..n())
                .take_while(|&y| !self.lines.is_drawn((tx, y).into(), Dir8::Lower))
                .find(|&y| self.marks().is_marked((tx, y).into()))
        }

        // Searches for the marked upper-right coordinate closest to the `coord`.
        pub fn search_closest_upper_right(&self, coord: CoordByU16) -> Option<CoordByU16> {
            if self.marks.all_upward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord.divide_u8();

            // Diagonal distance of the marked upper-right coordinate closest to the `coord`.
            let upper_right_dist = (1..=(n() - 1 - x).min(n() - 1 - y))
                .take_while(|&i| !self.lines.is_drawn((x + i, y + i).into(), Dir8::LowerLeft))
                .find(|&i| self.marks().is_marked((x + i, y + i).into()));

            upper_right_dist.map(|dist| (x + dist, y + dist).into())
        }

        // Searches for the marked lower-left coordinate closest to the `coord`.
        pub fn search_closest_lower_left(&self, coord: CoordByU16) -> Option<CoordByU16> {
            if self.marks.all_upward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord.divide_u8();

            // Diagonal distance of the marked lower-left coordinate closest to the `coord`.
            let lower_left_dist = (1..=x.min(y))
                .take_while(|&i| !self.lines.is_drawn((x - i, y - i).into(), Dir8::UpperRight))
                .find(|&i| self.marks().is_marked((x - i, y - i).into()));

            lower_left_dist.map(|dist| (x - dist, y - dist).into())
        }

        // Searches for the marked upper-left coordinate closest to the `coord`.
        pub fn search_closest_upper_left(&self, coord: CoordByU16) -> Option<CoordByU16> {
            if self.marks.all_downward_line_is_empty(coord) {
                return None;
            }

            let (x, y) = coord.divide_u8();

            // Diagonal distance of the marked upper-left coordinate closest to the `coord`.
            let upper_left_dist = (1..=x.min(n() - 1 - y))
                .take_while(|&i| !self.lines.is_drawn((x - i, y + i).into(), Dir8::LowerRight))
                .find(|&i| self.marks().is_marked((x - i, y + i).into()));

            upper_left_dist.map(|dist| (x - dist, y + dist).into())
        }

        // Searches for the marked lower_right coordinate closest to the `coord`.
        pub fn search_closest_lower_right(&self, coord: CoordByU16) -> Option<CoordByU16> {
            if self.marks.all_downward_line_is_empty(coord) {
                return None;
            }

            let (tx, ty) = coord.divide_u8();

            // Diagonal distance of the marked lower-right coordinate closest to the `coord`.
            let lower_right_dist = (1..=(n() - 1 - tx).min(ty))
                .take_while(|&i| {
                    !self
                        .lines
                        .is_drawn((tx + i, ty - i).into(), Dir8::UpperLeft)
                })
                .find(|&dist| self.marks().is_marked((tx + dist, ty - dist).into()));

            lower_right_dist.map(|dist| (tx + dist, ty - dist).into())
        }

        pub fn search_one_straight_selection(&self, tar_coord: CoordByU16) -> Option<Rect> {
            if self.marks().is_marked(tar_coord) {
                return None;
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = tar_coord.divide_u8();

            if self.marks.all_vertical_line_is_empty(tx)
                || self.marks.all_horizontal_line_is_empty(ty)
            {
                return None;
            }

            // x-coordinate of the marked left closest to the target coordinate.
            let left_x = self.search_closest_left(tar_coord);

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = self.search_closest_right(tar_coord);

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = self.search_closest_lower(tar_coord);

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = self.search_closest_upper(tar_coord);

            for (&adj_x, &adj_y) in [left_x, right_x]
                .iter()
                .cartesian_product([upper_y, lower_y].iter())
            {
                if let (Some(adj_x), Some(adj_y)) = (adj_x, adj_y) {
                    let opp_coord = CoordByU16::from((adj_x, adj_y));

                    if self.marks().is_marked(opp_coord)
                        && self.horizontal_line_is_empty((tx, adj_x), adj_y)
                        && self.vertical_line_is_empty(adj_x, (ty, adj_y))
                    {
                        return Some(Rect::Straight(RectCorners::new(
                            tar_coord,
                            (adj_x, ty).into(),
                            opp_coord,
                            (tx, adj_y).into(),
                        )));
                    }
                }
            }

            None
        }

        pub fn search_one_diagonal_selection(&self, tar_coord: CoordByU16) -> Option<Rect> {
            if self.marks().is_marked(tar_coord) {
                return None;
            }

            if self.marks.all_upward_line_is_empty(tar_coord)
                || self.marks.all_downward_line_is_empty(tar_coord)
            {
                return None;
            }

            // The upper-right coordinate of the marked grid point is closest to the `coord`.
            let upper_right_coord = self.search_closest_upper_right(tar_coord);

            // The lower-left coordinate of the marked grid point is closest to the `coord`.
            let lower_left_coord = self.search_closest_lower_left(tar_coord);

            // The upper-left coordinate of the marked grid point is closest to the `coord`.
            let upper_left_coord = self.search_closest_upper_left(tar_coord);

            // The lower-right coordinate of the marked grid point is closest to the `coord`.
            let lower_right_coord = self.search_closest_lower_right(tar_coord);

            let adj_coords = [
                upper_right_coord,
                upper_left_coord,
                lower_left_coord,
                lower_right_coord,
            ];

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = tar_coord.divide_u8();

            for i in 0..adj_coords.len() {
                if let (Some(adj_coord_1), Some(adj_coord_2)) =
                    (adj_coords[i], adj_coords[(i + 1) % adj_coords.len()])
                {
                    let (adj1_x, adj1_y) = adj_coord_1.divide_u8();
                    let (adj2_x, adj2_y) = adj_coord_2.divide_u8();

                    let opp_coord: CoordByU16 = if let (Some(opp_x), Some(opp_y)) = (
                        (adj1_x + adj2_x).checked_sub(tx),
                        (adj1_y + adj2_y).checked_sub(ty),
                    ) {
                        if opp_x < n() && opp_y < n() {
                            (opp_x, opp_y).into()
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    };

                    if self.marks().is_marked(opp_coord)
                        && self.diagonal_line_is_empty(adj_coord_1, opp_coord)
                        && self.diagonal_line_is_empty(adj_coord_2, opp_coord)
                    {
                        return Some(Rect::Diagonal(RectCorners::new(
                            tar_coord,
                            adj_coord_1,
                            opp_coord,
                            adj_coord_2,
                        )));
                    }
                }
            }

            // Returns `None` if the target coordinates cannot be marked.
            None
        }

        pub fn search_one_drawable_rectangle(&self, tar_coord: CoordByU16) -> Option<Rect> {
            if let Some(rect) = self.search_one_straight_selection(tar_coord) {
                return Some(rect);
            }

            if let Some(rect) = self.search_one_diagonal_selection(tar_coord) {
                return Some(rect);
            }

            None
        }

        fn count_markable(&self) -> usize {
            (0..(n() as usize).pow(2))
                .filter(|&coord_idx| {
                    self.search_one_drawable_rectangle(CoordByU16::from_coord_idx(coord_idx))
                        .is_some()
                })
                .count()
        }

        pub fn markable_num(&mut self) -> usize {
            if let Some(markable_num) = self.markable_num {
                return markable_num;
            }

            let markable_num = self.count_markable();

            self.markable_num = Some(markable_num);

            markable_num
        }

        pub fn search_straight_selections(&self, tar_coord: CoordByU16) -> Vec<Rect> {
            if self.marks().is_marked(tar_coord) {
                return vec![];
            }

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = tar_coord.divide_u8();

            if self.marks.all_vertical_line_is_empty(tx)
                || self.marks.all_horizontal_line_is_empty(ty)
            {
                return vec![];
            }

            // x-coordinate of the marked left closest to the target coordinate.
            let left_x = self.search_closest_left(tar_coord);

            // x-coordinate of the marked right closest to the target coordinate.
            let right_x = self.search_closest_right(tar_coord);

            // y-coordinate of the marked lower closest to the target coordinate.
            let lower_y = self.search_closest_lower(tar_coord);

            // y-coordinate of the marked upper closest to the target coordinate.
            let upper_y = self.search_closest_upper(tar_coord);

            [left_x, right_x]
                .iter()
                .cartesian_product([upper_y, lower_y].iter())
                .filter_map(|(&other_x, &other_y)| {
                    if let (Some(adj_x), Some(adj_y)) = (other_x, other_y) {
                        let opp_coord = CoordByU16::from((adj_x, adj_y));

                        if self.marks().is_marked(opp_coord)
                            && self.horizontal_line_is_empty((tx, adj_x), adj_y)
                            && self.vertical_line_is_empty(adj_x, (ty, adj_y))
                        {
                            return Some(Rect::Straight(RectCorners::new(
                                tar_coord,
                                (adj_x, ty).into(),
                                opp_coord,
                                (tx, adj_y).into(),
                            )));
                        }
                    }

                    None
                })
                .collect()
        }

        pub fn search_diagonal_selections(&self, coord: CoordByU16) -> Vec<Rect> {
            if self.marks().is_marked(coord) {
                return vec![];
            }

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

            let adj_coords = [
                upper_right_coord,
                upper_left_coord,
                lower_left_coord,
                lower_right_coord,
            ];

            // Target x-coordinate and y-coordinate.
            let (tx, ty) = coord.divide_u8();

            (0..adj_coords.len())
                .filter_map(|i| {
                    if let (Some(adj_coord_1), Some(adj_coord_2)) =
                        (adj_coords[i], adj_coords[(i + 1) % adj_coords.len()])
                    {
                        let (adj1_x, adj1_y) = adj_coord_1.divide_u8();
                        let (adj2_x, ajd2_y) = adj_coord_2.divide_u8();

                        let opp_coord: CoordByU16 = if let (Some(opp_x), Some(opp_y)) = (
                            (adj1_x + adj2_x).checked_sub(tx),
                            (adj1_y + ajd2_y).checked_sub(ty),
                        ) {
                            if opp_x < n() && opp_y < n() {
                                (opp_x, opp_y).into()
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        };

                        if self.marks().is_marked(opp_coord)
                            && self.diagonal_line_is_empty(adj_coord_1, opp_coord)
                            && self.diagonal_line_is_empty(adj_coord_2, opp_coord)
                        {
                            return Some(Rect::Diagonal(RectCorners::new(
                                coord,
                                adj_coord_1,
                                opp_coord,
                                adj_coord_2,
                            )));
                        }
                    }

                    None
                })
                .collect()
        }

        pub fn search_selections(&self, coord: CoordByU16) -> Vec<Rect> {
            let mut straight_selections = self.search_straight_selections(coord);
            let mut diagonal_selections = self.search_diagonal_selections(coord);
            straight_selections.append(&mut diagonal_selections);

            straight_selections
        }

        // pub fn general_possible_straight_selections(&self) -> Vec<Selection> {
        //     (0..n())
        //         .flat_map(|x| {
        //             (0..n()).filter_map(move |y| self.search_one_straight_selection((x, y).into()))
        //         })
        //         .collect()
        // }

        // pub fn general_possible_diagonal_selections(&self) -> Vec<Selection> {
        //     (0..n())
        //         .flat_map(|x| {
        //             (0..n()).filter_map(move |y| self.search_one_diagonal_selection((x, y).into()))
        //         })
        //         .collect()
        // }

        // pub fn general_possible_selections(&self) -> Vec<Selection> {
        //     (0..n())
        //         .flat_map(|x| {
        //             (0..n()).filter_map(move |y| self.search_one_selection((x, y).into()))
        //         })
        //         .collect()
        // }

        pub fn all_drawable_straight_rectangles(&self) -> Vec<Rect> {
            (0..(n() as usize).pow(2))
                .flat_map(|coord_idx| {
                    self.search_straight_selections(CoordByU16::from_coord_idx(coord_idx))
                })
                .collect()
        }

        pub fn all_drawable_diagonal_rectangles(&self) -> Vec<Rect> {
            (0..(n() as usize).pow(2))
                .flat_map(|coord_idx| {
                    self.search_diagonal_selections(CoordByU16::from_coord_idx(coord_idx))
                })
                .collect()
        }

        pub fn all_drawable_rectangles(&self) -> Vec<Rect> {
            self.all_drawable_straight_rectangles()
                .into_iter()
                .chain(self.all_drawable_diagonal_rectangles().into_iter())
                .collect()
        }

        fn draw_vertical_grid_line(
            &mut self,
            x: CoordElemByU8,
            y_pair: (CoordElemByU8, CoordElemByU8),
        ) {
            let (mut y1, mut y2) = y_pair;
            if y1 > y2 {
                mem::swap(&mut y1, &mut y2)
            }

            for y in y1..y2 {
                self.lines.draw_line((x, y).into(), Dir8::Upper);
            }
        }

        fn draw_horizontal_grid_line(
            &mut self,
            x_pair: (CoordElemByU8, CoordElemByU8),
            y: CoordElemByU8,
        ) {
            let (mut x1, mut x2) = x_pair;
            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
            }

            for x in x1..x2 {
                self.lines.draw_line((x, y).into(), Dir8::Right);
            }
        }

        #[allow(dead_code)]
        fn draw_straight_grid_line(&mut self, coord1: CoordByU16, coord2: CoordByU16) {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.draw_vertical_grid_line(x1, (y1, y2));
            } else if y1 == y2 {
                self.draw_horizontal_grid_line((x1, x2), y1);
            } else {
                panic!("Either the x or y coordinate must be the same.");
            }
        }

        fn draw_upward_grid_line(&mut self, source_coord: CoordByU16, dist: CoordElemByU8) {
            let (sx, sy) = source_coord.divide_u8();

            for i in 0..dist {
                self.lines
                    .draw_line((sx + i, sy + i).into(), Dir8::UpperRight);
            }
        }

        fn draw_downward_grid_line(&mut self, source_coord: CoordByU16, dist: CoordElemByU8) {
            let (sx, sy) = source_coord.divide_u8();

            for i in 0..dist {
                self.lines
                    .draw_line((sx + i, sy - i).into(), Dir8::LowerRight);
            }
        }

        fn draw_diagonal_grid_line(&mut self, coord1: CoordByU16, coord2: CoordByU16) {
            let (mut x1, mut y1) = coord1.divide_u8();
            let (mut x2, mut y2) = coord2.divide_u8();

            if x1 > x2 {
                mem::swap(&mut x1, &mut x2);
                mem::swap(&mut y1, &mut y2);
            }

            let dist = x2 - x1;

            if y1 < y2 {
                debug_assert_eq!(y2 - y1, dist);

                self.draw_upward_grid_line((x1, y1).into(), dist);
            } else {
                debug_assert_eq!(y1 - y2, dist);

                self.draw_downward_grid_line((x1, y1).into(), dist);
            }
        }

        fn draw_grid_line(&mut self, coord1: CoordByU16, coord2: CoordByU16) {
            let (x1, y1) = coord1.divide_u8();
            let (x2, y2) = coord2.divide_u8();

            if x1 == x2 {
                self.draw_vertical_grid_line(x1, (y1, y2));
            } else if y1 == y2 {
                self.draw_horizontal_grid_line((x1, x2), y1);
            } else {
                self.draw_diagonal_grid_line(coord1, coord2);
            }
        }

        pub fn selectable(&self, rect: &Rect) -> bool {
            self.search_one_drawable_rectangle(rect.target()).is_some()
        }

        fn marking(&mut self, rect: &Rect) {
            debug_assert!(self.selectable(rect));

            self.marks.add_mark(rect.target());

            let corner_coords = rect.corner_coords();

            for i in 0..corner_coords.len() {
                self.draw_grid_line(
                    corner_coords[i],
                    corner_coords[(i + 1) % corner_coords.len()],
                );
            }
        }
    }

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct RectJoinNode {
        grid_state: GridState,
        history: Vec<Rect>,
    }

    impl RectJoinNode {
        pub fn new(xy: &Vec<CoordByU16>) -> Self {
            Self {
                grid_state: GridState::new(xy),
                history: vec![],
            }
        }

        pub fn grid_state(&self) -> &GridState {
            &self.grid_state
        }

        pub fn grid_state_mut(&mut self) -> &mut GridState {
            &mut self.grid_state
        }

        pub fn history(&self) -> &Vec<Rect> {
            &self.history
        }

        pub fn marking(&mut self, rect: Rect) {
            self.grid_state.marking(&rect);
            self.history.push(rect);
        }

        pub fn try_marking(&mut self, rect: Rect) -> bool {
            if !self.grid_state().selectable(&rect) {
                return false;
            }

            self.marking(rect);

            true
        }

        pub fn marked_node(&self, rect: Rect) -> Self {
            let mut marked_node = self.clone();
            marked_node.marking(rect);

            marked_node
        }

        pub fn show_history(&self) {
            println!("{}", self.history.len());

            for rect in &self.history {
                rect.show();
            }
        }

        pub fn moved_num(&self) -> usize {
            self.history.len()
        }

        pub fn sum_weight(&self) -> WeightByU32 {
            self.grid_state().marks().sum_weight()
        }

        pub fn score(&self) -> ScoreByU32 {
            self.grid_state().marks().score()
        }
    }

    pub mod cmp_node {
        use std::{cmp::Ordering, fmt::Debug};

        use super::{initial_weight, RectJoinNode};

        pub trait CmpNode:
            Debug + Clone + From<RectJoinNode> + PartialEq + Eq + PartialOrd + Ord
        {
            fn cmp_node(&self, other: &Self) -> Ordering;

            fn into_node(self) -> RectJoinNode;
        }

        #[derive(Debug, Clone)]
        pub struct CmpBySumWeight(pub RectJoinNode);

        impl CmpNode for CmpBySumWeight {
            fn cmp_node(&self, other: &Self) -> Ordering {
                self.0
                    .grid_state()
                    .marks()
                    .sum_weight()
                    .cmp(&other.0.grid_state().marks().sum_weight())
            }

            fn into_node(self) -> RectJoinNode {
                self.0
            }
        }

        impl From<RectJoinNode> for CmpBySumWeight {
            fn from(node: RectJoinNode) -> Self {
                Self(node)
            }
        }

        impl PartialEq for CmpBySumWeight {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl Eq for CmpBySumWeight {}

        impl PartialOrd for CmpBySumWeight {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CmpBySumWeight {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        #[derive(Debug, Clone)]
        pub struct CmpByRevAvgLen(pub RectJoinNode);

        impl CmpNode for CmpByRevAvgLen {
            fn cmp_node(&self, other: &Self) -> Ordering {
                let self_avg_len =
                    self.0.grid_state().lines().drawn_len() as f64 / self.0.moved_num() as f64;
                let other_avg_len =
                    other.0.grid_state().lines().drawn_len() as f64 / other.0.moved_num() as f64;

                self_avg_len.partial_cmp(&other_avg_len).unwrap().reverse()
            }

            fn into_node(self) -> RectJoinNode {
                self.0
            }
        }

        impl PartialEq for CmpByRevAvgLen {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl From<RectJoinNode> for CmpByRevAvgLen {
            fn from(node: RectJoinNode) -> Self {
                Self(node)
            }
        }

        impl Eq for CmpByRevAvgLen {}

        impl PartialOrd for CmpByRevAvgLen {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CmpByRevAvgLen {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        #[derive(Debug, Clone)]
        pub struct CmpByInterval(pub RectJoinNode);

        impl CmpByInterval {
            fn cmp_by_avg_inc_weight(&self, other: &Self) -> Ordering {
                let self_avg = (self.0.grid_state().marks().sum_weight() - initial_weight()) as f64
                    / (self.0.moved_num() + 1) as f64;
                let other_avg = (other.0.grid_state().marks().sum_weight() - initial_weight())
                    as f64
                    / (other.0.moved_num() + 1) as f64;

                self_avg.partial_cmp(&other_avg).unwrap().reverse()
            }
        }

        impl CmpNode for CmpByInterval {
            fn cmp_node(&self, other: &Self) -> Ordering {
                let self_x_interval = self.0.grid_state().marks().x_interval();
                let self_y_interval = self.0.grid_state().marks().y_interval();
                let self_add_interval = self_x_interval + self_y_interval;

                let other_x_interval = other.0.grid_state().marks().x_interval();
                let other_y_interval = other.0.grid_state().marks().y_interval();
                let other_add_interval = other_x_interval + other_y_interval;

                match self_add_interval.cmp(&other_add_interval) {
                    Ordering::Less => Ordering::Less,
                    Ordering::Equal => self.cmp_by_avg_inc_weight(other),
                    Ordering::Greater => Ordering::Greater,
                }
            }

            fn into_node(self) -> RectJoinNode {
                self.0
            }
        }

        impl From<RectJoinNode> for CmpByInterval {
            fn from(node: RectJoinNode) -> Self {
                Self(node)
            }
        }

        impl PartialEq for CmpByInterval {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl Eq for CmpByInterval {}

        impl PartialOrd for CmpByInterval {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CmpByInterval {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        #[derive(Debug, Clone)]
        pub struct CmpByMarkableNum(pub RectJoinNode, pub usize);

        impl CmpNode for CmpByMarkableNum {
            fn cmp_node(&self, other: &Self) -> Ordering {
                self.1.cmp(&other.1)
            }

            fn into_node(self) -> RectJoinNode {
                self.0
            }
        }

        impl From<RectJoinNode> for CmpByMarkableNum {
            fn from(mut node: RectJoinNode) -> Self {
                let markable_num = node.grid_state_mut().markable_num();

                Self(node, markable_num)
            }
        }

        impl PartialEq for CmpByMarkableNum {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl Eq for CmpByMarkableNum {}

        impl PartialOrd for CmpByMarkableNum {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CmpByMarkableNum {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        #[derive(Debug, Clone)]
        pub struct CmpByRevDrawnLen(pub RectJoinNode);

        impl CmpNode for CmpByRevDrawnLen {
            fn cmp_node(&self, other: &Self) -> Ordering {
                self.0
                    .grid_state()
                    .lines()
                    .drawn_len()
                    .cmp(&other.0.grid_state().lines().drawn_len())
                    .reverse()
            }

            fn into_node(self) -> RectJoinNode {
                self.0
            }
        }

        impl From<RectJoinNode> for CmpByRevDrawnLen {
            fn from(node: RectJoinNode) -> Self {
                Self(node)
            }
        }

        impl PartialEq for CmpByRevDrawnLen {
            fn eq(&self, other: &Self) -> bool {
                self.cmp_node(other) == Ordering::Equal
            }
        }

        impl Eq for CmpByRevDrawnLen {}

        impl PartialOrd for CmpByRevDrawnLen {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp_node(other))
            }
        }

        impl Ord for CmpByRevDrawnLen {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
    }
}
