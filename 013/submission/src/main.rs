use std::{collections::BinaryHeap, time::Instant};

use rand::{thread_rng, seq::SliceRandom};

use crate::module::StatusTree;

/// Data type of coordinate
type Coord = (usize, usize);

/// Data type of computer maps represented by characters
type MapByChars = Vec<Vec<char>>;

static TIME_LIMIT: f64 = 2.8;

/// Length of one side of the grid
static mut N: Option<usize> = None;

/// Number of computer types
static mut K: Option<usize> = None;

fn main() {
    initialize_start_instant();

    ////////////////////////////////////////////////////////////////////////////////
    // Reads from standard input.
    // `n` - Length of one side of the grid
    // `k` - Number of computer types
    let (n, k) = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        (
            iter.next().unwrap().parse::<usize>().unwrap(),
            iter.next().unwrap().parse::<usize>().unwrap(),
        )
    };
    // Initial computer map represented by characters
    let mut init_computer_map: MapByChars = vec![];
    for _ in 0..n {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        init_computer_map.push(line.trim().chars().collect());
    }

    // Initializes global variables.
    initialize_global_variables(n, k);
    ////////////////////////////////////////////////////////////////////////////////

    if k <= 2 {
        strategy1(init_computer_map);
    } else {
        strategy2(init_computer_map);
    }
}

fn strategy1(init_computer_map: MapByChars) {
    static CHILDREN_NUM: usize = 50;

    let mut rng = thread_rng();

    let mut status_tree = StatusTree::new(init_computer_map);
    let mut heap = BinaryHeap::from(vec![(status_tree.get_node(0).score(), 0)]);

    while let Some((_, node_idx)) = heap.pop() {
        if status_tree.get_node(node_idx).rem_turn() == 0 {
            continue;
        }

        for _ in 0..CHILDREN_NUM {
            let (coord, moveable_dirs) = status_tree
                .get_node(node_idx)
                .random_select_movable_computer(&mut rng);
            let dir = *moveable_dirs.choose(&mut rng).unwrap();
            let moved_node_idx = status_tree.add_one_moved_node(node_idx, coord, dir);
 
            heap.push((status_tree.get_node(moved_node_idx).score(), moved_node_idx));
        }
 
        if get_elapsed_time() >= TIME_LIMIT {
            break;
        }
    }
 
    let best_node = status_tree.find_best_node();
    status_tree.show_node_solution(best_node);
    eprintln!("Score = {}", best_node.score());
}

fn strategy2(init_computer_map: MapByChars) {
    static CHILDREN_NUM: usize = 15;
    static MAX_WARP_DIST: usize = 5;

    let mut rng = thread_rng();

    let mut status_tree = StatusTree::new(init_computer_map);
    let mut heap = BinaryHeap::from(vec![(status_tree.get_node(0).score(), 0)]);

    while let Some((_, node_idx)) = heap.pop() {
        if status_tree.get_node(node_idx).rem_turn() == 0 {
            continue;
        }
 
        for _ in 0..CHILDREN_NUM {
            let max_warp_dist = MAX_WARP_DIST.min(status_tree.get_node(node_idx).rem_turn());
            let moved_node_idx = status_tree.add_random_moved_node(node_idx, max_warp_dist, &mut rng);
            heap.push((status_tree.get_node(moved_node_idx).score(), moved_node_idx));
        }
 
        if get_elapsed_time() >= TIME_LIMIT {
            break;
        }
    }
 
    let best_node = status_tree.find_best_node();
    status_tree.show_node_solution(best_node);
    eprintln!("Score = {}", best_node.score());
}

#[allow(dead_code)]
fn strategy3(init_computer_map: MapByChars) {
    static CHILDREN_NUM: usize = 15;
    static MAX_WARP_DIST: usize = 5;

    let mut rng = thread_rng();

    let mut status_tree = StatusTree::new(init_computer_map);
    let mut beam = vec![BinaryHeap::from(vec![(status_tree.get_node(0).score(), 0)])];

    while get_elapsed_time() <= TIME_LIMIT {
        for depth in 0..beam.len() {
            if get_elapsed_time() > TIME_LIMIT {
                break;
            }

            if let Some((_, node_idx)) = beam[depth].pop() {
                if get_elapsed_time() > TIME_LIMIT {
                    break;
                }

                if status_tree.get_node(node_idx).rem_turn() == 0 {
                    continue;
                }

                if depth + 1 == beam.len() {
                    beam.push(BinaryHeap::new());
                }

                for _ in 0..CHILDREN_NUM {
                    let max_warp_dist = MAX_WARP_DIST.min(status_tree.get_node(node_idx).rem_turn());
                    let moved_node_idx = status_tree.add_random_moved_node(node_idx, max_warp_dist, &mut rng);
                    beam[depth + 1].push((status_tree.get_node(moved_node_idx).score(), moved_node_idx));
                }

                beam[depth].push((0, node_idx));
            }
        }
    }

    let best_node = status_tree.find_best_node();
    status_tree.show_node_solution(best_node);
    eprintln!("Score = {}", best_node.score());
}

/// Initializes global variables.
fn initialize_global_variables(n: usize, k: usize) {
    unsafe {
        assert!(N.is_none() && K.is_none(), "Already initialized.");

        N = Some(n);
        K = Some(k);
    }
}

/// Gets the length of one side of the grid.
pub fn get_n() -> usize {
    unsafe { N.unwrap() }
}

/// Gets the number of computer types.
pub fn get_k() -> usize {
    unsafe { K.unwrap() }
}

pub fn get_max_turn() -> usize {
    100 * get_k()
}

/// Checks if the `coord` is on the grid.
pub fn within_grid(coord: Coord) -> bool {
    let n = get_n();
    let (x, y) = coord;

    x < n && y < n
}

pub fn assert_within_grid(coord: Coord) {
    let (x, y) = coord;

    assert!(
        within_grid(coord),
        "The coordinate ({0}, {1}) is out of range of the grid.\
        Grid size is {2}x{2}",
        x,
        y,
        get_n(),
    );
}

fn get_start_time_instant() -> Instant {
    static mut START_TIME_INSTANT: Option<Instant> = None;
    unsafe {
        match START_TIME_INSTANT {
            Some(v) => v,
            None => {
                let v = Instant::now();
                START_TIME_INSTANT = Some(v);
                v
            }
        }
    }
}

fn initialize_start_instant() {
    get_start_time_instant();
}

fn get_elapsed_time() -> f64 {
    let curr_time_instant = Instant::now();
    let duration = curr_time_instant.duration_since(get_start_time_instant());
    duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9
}

pub mod module {
    use std::{
        char,
        cmp::Reverse,
        collections::VecDeque,
        hash::Hash,
        ops::RangeBounds,
    };

    use rand::{Rng, seq::SliceRandom};

    use crate::{
        assert_within_grid,
        atcoder8_library::{direction::Dir, union_find::UnionFind},
        get_k, get_max_turn, get_n, within_grid, Coord, MapByChars,
    };

    fn min_max<T>(x1: T, x2: T) -> (T, T)
    where
        T: Ord,
    {
        if x1 <= x2 {
            (x1, x2)
        } else {
            (x2, x1)
        }
    }

    trait ToCoord {
        fn to_coord(self) -> Coord;
    }

    impl ToCoord for usize {
        fn to_coord(self) -> Coord {
            let n = get_n();
            let coord = (self / n, self % n);

            assert_within_grid(coord);

            coord
        }
    }

    trait ToCellIdx {
        fn to_cell_idx(self) -> usize;
    }

    impl ToCellIdx for Coord {
        fn to_cell_idx(self) -> usize {
            let (x, y) = self;
            x * get_n() + y
        }
    }

    trait Moved {
        fn moved(self, dir: Dir) -> Self;
    }

    impl Moved for Coord {
        fn moved(self, dir: Dir) -> Self {
            assert!(
                within_grid(self),
                "The specified coordinate is out of range of the grid."
            );

            let n = get_n();
            let (x, y) = self;

            match dir {
                Dir::Up => {
                    assert_ne!(x, 0, "The upper cell does not exist.");
                    (x - 1, y)
                }
                Dir::Down => {
                    assert_ne!(x, n - 1, "The lower cell does not exist.");
                    (x + 1, y)
                }
                Dir::Left => {
                    assert_ne!(y, 0, "The left cell does not exist.");
                    (x, y - 1)
                }
                Dir::Right => {
                    assert_ne!(y, n - 1, "The right cell does not exist.");
                    (x, y + 1)
                }
            }
        }
    }

    fn start_and_end<R>(rng: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let start = match rng.start_bound() {
            std::ops::Bound::Included(&start_bound) => start_bound,
            std::ops::Bound::Excluded(&start_bound) => start_bound + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match rng.end_bound() {
            std::ops::Bound::Included(&end_bound) => end_bound + 1,
            std::ops::Bound::Excluded(&end_bound) => end_bound,
            std::ops::Bound::Unbounded => get_n(),
        };

        assert!(
            start <= end,
            "The slice index start at {} but end at {}.",
            start,
            end
        );

        let n = get_n();

        assert!(
            end <= n,
            "The specified range {}..{} is outside the range of the sequence; the length of sequence is {}.",
            start,
            end,
            n,
        );

        (start, end)
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Bitboard(Vec<usize>);

    impl Bitboard {
        pub fn new() -> Self {
            Self(vec![0; get_n()])
        }

        pub fn get(&self, coord: Coord) -> bool {
            assert_within_grid(coord);

            let (x, y) = coord;
            (self.0[x] >> y) & 1 == 1
        }

        pub fn set(&mut self, coord: Coord, value: bool) -> bool {
            if self.get(coord) == value {
                return false;
            }

            let (x, y) = coord;
            self.0[x] ^= 1 << y;

            true
        }

        pub fn count_ones<R1, R2>(&self, x_rng: R1, y_rng: R2) -> usize
        where
            R1: RangeBounds<usize>,
            R2: RangeBounds<usize>,
        {
            let (start_x, end_x) = start_and_end(x_rng);
            let (start_y, end_y) = start_and_end(y_rng);
            let mask = (1 << end_y) - (1 << start_y);

            (start_x..end_x)
                .map(|x| (self.0[x] & mask).count_ones() as usize)
                .sum()
        }

        pub fn any_rect<R1, R2>(&self, x_rng: R1, y_rng: R2) -> bool
        where
            R1: RangeBounds<usize>,
            R2: RangeBounds<usize>,
        {
            let (start_x, end_x) = start_and_end(x_rng);
            let (start_y, end_y) = start_and_end(y_rng);
            let mask = (1 << end_y) - (1 << start_y);

            (start_x..end_x).any(|x| self.0[x] & mask != 0)
        }

        pub fn all_rect<R1, R2>(&self, x_rng: R1, y_rng: R2) -> bool
        where
            R1: RangeBounds<usize>,
            R2: RangeBounds<usize>,
        {
            let (start_x, end_x) = start_and_end(x_rng);
            let (start_y, end_y) = start_and_end(y_rng);
            let mask = (1 << end_y) - (1 << start_y);

            (start_x..end_x).all(|x| self.0[x] == mask)
        }

        pub fn set_rect<R1, R2>(&mut self, x_rng: R1, y_rng: R2, value: bool)
        where
            R1: RangeBounds<usize>,
            R2: RangeBounds<usize>,
        {
            let (start_x, end_x) = start_and_end(x_rng);
            let (start_y, end_y) = start_and_end(y_rng);

            if value {
                let mask = (1 << end_y) - (1 << start_y);
                (start_x..end_x).for_each(|x| self.0[x] |= mask);
            } else {
                let mask = !((1 << end_y) - (1 << start_y));
                (start_x..end_x).for_each(|x| self.0[x] &= mask);
            }
        }
    }

    impl Default for Bitboard {
        fn default() -> Self {
            Self::new()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct ServerRoom(Vec<Bitboard>);

    impl ServerRoom {
        /// Gets the type of computer at the specified coordinate.
        /// If the computer does not exist, returns `None`.
        pub fn computer_type(&self, coord: Coord) -> Option<usize> {
            assert_within_grid(coord);

            for (i, bitboard) in self.0.iter().enumerate() {
                if bitboard.get(coord) {
                    return Some(i);
                }
            }

            None
        }

        /// Check if the computer is placed at the specified coordinates.
        pub fn filled(&self, coord: Coord) -> bool {
            assert!(
                within_grid(coord),
                "The specified coordinate is out of range of the grid."
            );

            self.computer_type(coord).is_some()
        }

        pub fn count_ones<R1, R2>(&self, x_rng: R1, y_rng: R2) -> usize
        where
            R1: RangeBounds<usize> + Clone,
            R2: RangeBounds<usize> + Clone,
        {
            self.0
                .iter()
                .map(|bitboard| bitboard.count_ones(x_rng.clone(), y_rng.clone()))
                .sum()
        }

        pub fn any_rect<R1, R2>(&self, x_rng: R1, y_rng: R2) -> bool
        where
            R1: RangeBounds<usize> + Clone,
            R2: RangeBounds<usize> + Clone,
        {
            self.0
                .iter()
                .any(|bitboard| bitboard.any_rect(x_rng.clone(), y_rng.clone()))
        }

        pub fn all_rect<R1, R2>(&self, x_rng: R1, y_rng: R2) -> bool
        where
            R1: RangeBounds<usize> + Clone,
            R2: RangeBounds<usize> + Clone,
        {
            self.0
                .iter()
                .all(|bitboard| bitboard.all_rect(x_rng.clone(), y_rng.clone()))
        }

        /// Checks if the specified move is possible.
        pub fn moveable(&self, coord: Coord, dir: Dir) -> bool {
            let n = get_n();
            let (x, y) = coord;

            let within_grid_flag = match dir {
                Dir::Up => x > 0,
                Dir::Down => x < n - 1,
                Dir::Left => y > 0,
                Dir::Right => y < n - 1,
            };

            within_grid_flag && self.filled(coord) && !self.filled(coord.moved(dir))
        }

        /// Move one computer one time.
        pub fn move_one(&mut self, coord: Coord, dir: Dir) {
            assert!(
                self.moveable(coord, dir),
                "The specified move is not possible."
            );

            let computer_type = self.computer_type(coord).unwrap();
            self.0[computer_type].set(coord, false);
            self.0[computer_type].set(coord.moved(dir), true);
        }

        /// Gets the symbol corresponding to the cell.
        fn cell_symbol(&self, coord: Coord, one_based: bool, empty_symbol: char) -> char {
            let computer_type = self.computer_type(coord);
            if let Some(computer_type) = computer_type {
                char::from_digit(computer_type as u32 + if one_based { 1 } else { 0 }, 10).unwrap()
            } else {
                empty_symbol
            }
        }

        pub fn random_move_route<R>(&self, source_coord: Coord, max_dist: usize, rng: &mut R) -> Option<Vec<Dir>>
        where
            R: Rng,
        {
            assert!(self.filled(source_coord));
            assert_ne!(max_dist, 0);

            let n = get_n();
            let mut dp = vec![None; n.pow(2)];
            let mut que = VecDeque::from(vec![(source_coord, 0)]);
            let mut dest_coords = vec![];

            while let Some((curr_coord, curr_dist)) = que.pop_front() {
                Dir::all_directions().into_iter().for_each(|dir| {
                    if self.moveable(curr_coord, dir) {
                        let next_coord = curr_coord.moved(dir);
                        if dp[next_coord.to_cell_idx()].is_none() {
                            dp[next_coord.to_cell_idx()] = Some(dir);
                            dest_coords.push(next_coord);
                            if curr_dist < max_dist {
                                que.push_back((next_coord, curr_dist + 1));
                            }
                        }
                    }
                });
            }

            let dest_coord = if let Some(dest_coord) = dest_coords.choose(rng) {
                *dest_coord
            } else {
                return None;
            };

            let mut move_dirs = vec![];
            let mut curr_coord = dest_coord;

            while curr_coord != source_coord {
                let dir = dp[curr_coord.to_cell_idx()].unwrap();
                move_dirs.push(dir);
                curr_coord = curr_coord.moved(dir.opposite());
            }

            move_dirs.reverse();

            Some(move_dirs)
        }

        pub fn search_move_route(&self, source_coord: Coord, dest_coord: Coord) -> Option<Vec<Dir>> {
            assert!(self.filled(source_coord) && !self.filled(dest_coord));

            let n = get_n();
            let mut dp = vec![None; n.pow(2)];
            let mut que = VecDeque::from(vec![source_coord]);

            if source_coord == dest_coord {
                return Some(vec![]);
            }

            if !self.filled(source_coord) || self.filled(dest_coord) {
                return None;
            }

            while let Some(curr_coord) = que.pop_front() {
                Dir::all_directions().into_iter().for_each(|dir| {
                    if self.moveable(curr_coord, dir) {
                        let next_coord = curr_coord.moved(dir);
                        if dp[next_coord.to_cell_idx()].is_none() {
                            dp[next_coord.to_cell_idx()] = Some(dir);
                            que.push_back(next_coord);
                        }
                    }
                });
            }

            let mut move_dirs = vec![];
            let mut curr_coord = dest_coord;

            while curr_coord != source_coord {
                let dir = dp[curr_coord.to_cell_idx()].unwrap();
                move_dirs.push(dir);
                curr_coord = curr_coord.moved(dir.opposite());
            }

            move_dirs.reverse();

            Some(move_dirs)
        }

        /// Shows the status of the server room.
        pub fn show_with_config(&self, one_based: bool, empty_symbol: char) {
            let n = get_n();

            for x in 0..n {
                for y in 0..n {
                    print!("{}", self.cell_symbol((x, y), one_based, empty_symbol));
                }
                println!();
            }
        }

        /// Shows the status of the server room.
        pub fn show(&self) {
            self.show_with_config(false, '.');
        }

        pub fn moveable_directions(&self, coord: Coord) -> Vec<Dir> {
            Dir::all_directions()
                .into_iter()
                .filter(|&dir| self.moveable(coord, dir))
                .collect()
        }

        pub fn random_select_computer<R>(&self, rng: &mut R) -> Coord
        where
            R: Rng,
        {
            let n = get_n();
            let pow2_n = n.pow(2);

            let mut cell_idx = rng.gen_range(0, pow2_n);
            loop {
                let coord = cell_idx.to_coord();
                if self.filled(coord) {
                    return coord;
                }
                cell_idx = (cell_idx + 1) % pow2_n;
            }
        }

        pub fn random_select_movable_computer<R>(&self, rng: &mut R) -> (Coord, Vec<Dir>)
        where
            R: Rng,
        {
            let n = get_n();

            let mut cell_idx = rng.gen_range(0, n.pow(2));

            loop {
                let coord = cell_idx.to_coord();
                let moveable_dirs = self.moveable_directions(coord);
                if !moveable_dirs.is_empty() {
                    return (coord, moveable_dirs);
                }
                cell_idx = (cell_idx + 1) % n.pow(2);
            }
        }

        pub fn directly_connectable(&self, coord1: Coord, coord2: Coord) -> bool {
            assert_ne!(
                coord1, coord2,
                "The coordinates of `coord1` and `coord2` are the same."
            );

            if !self.filled(coord1) || !self.filled(coord2) {
                return false;
            }

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                let (min_y, max_y) = min_max(y1, y2);
                !self.any_rect(x1..=x1, (min_y + 1)..max_y)
            } else if y1 == y2 {
                let (min_x, max_x) = min_max(x1, x2);
                !self.any_rect((min_x + 1)..max_x, y1..=y1)
            } else {
                false
            }
        }

        pub fn sandwiched_computers(&self, coord1: Coord, coord2: Coord) -> Vec<Coord> {
            assert_ne!(
                coord1, coord2,
                "The coordinates of `coord1` and `coord2` are the same."
            );

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            assert!(
                self.filled(coord1),
                "No computer exists at ({} {}).",
                x1,
                y1
            );
            assert!(
                self.filled(coord2),
                "No computer exists at ({} {}).",
                x2,
                y2
            );

            let (min_x, max_x) = min_max(x1, x2);
            let (min_y, max_y) = min_max(y1, y2);

            if x1 == x2 || y1 == y2 {
                let computer_series: Vec<Coord> = if x1 == x2 {
                    ((min_y + 1)..max_y)
                        .map(|y| (x1, y))
                        .filter(|&coord| self.filled(coord))
                        .collect()
                } else {
                    ((min_x + 1)..max_x)
                        .map(|x| (x, y1))
                        .filter(|&coord| self.filled(coord))
                        .collect()
                };

                if computer_series.len() == 1 {
                    return vec![computer_series[0]];
                } else {
                    return vec![];
                }
            }

            let x_rng = (min_x + 1)..max_x;
            let y_rng = (min_y + 1)..max_y;

            let mut computers = vec![];

            if self.filled((x1, y2))
                && !self.any_rect(x1..=x1, y_rng.clone())
                && !self.any_rect(x_rng.clone(), y2..=y2)
            {
                computers.push((x1, y2));
            }

            if self.filled((x2, y1))
                && !self.any_rect(x_rng, y1..=y1)
                && !self.any_rect(x2..=x2, y_rng)
            {
                computers.push((x2, y1));
            }

            computers
        }

        pub fn find_way_to_connect_same_type(&self) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
            let n = get_n();
            let mut cell_graph = vec![vec![]; n.pow(2)];
            let mut uf = UnionFind::new(n.pow(2));

            // Horizontal search
            for x in 0..n {
                let mut computer_types = vec![];

                for y in 0..n {
                    if let Some(computer_type) = self.computer_type((x, y)) {
                        computer_types.push((y, computer_type));
                    }
                }

                for window in computer_types.windows(2) {
                    let (y1, computer_type_1) = window[0];
                    let (y2, computer_type_2) = window[1];

                    let (cell_idx_1, cell_idx_2) = ((x, y1).to_cell_idx(), (x, y2).to_cell_idx());

                    if computer_type_1 == computer_type_2 {
                        cell_graph[cell_idx_1].push(cell_idx_2);
                        cell_graph[cell_idx_2].push(cell_idx_1);
                        uf.merge(cell_idx_1, cell_idx_2);
                    }
                }
            }

            // Vertical search
            for y in 0..n {
                let mut computer_types = vec![];

                for x in 0..n {
                    if let Some(computer_type) = self.computer_type((x, y)) {
                        computer_types.push((x, computer_type));
                    }
                }

                for window in computer_types.windows(2) {
                    let (x1, computer_type_1) = window[0];
                    let (x2, computer_type_2) = window[1];

                    let (cell_idx_1, cell_idx_2) = ((x1, y).to_cell_idx(), (x2, y).to_cell_idx());

                    if computer_type_1 == computer_type_2 {
                        cell_graph[cell_idx_1].push(cell_idx_2);
                        cell_graph[cell_idx_2].push(cell_idx_1);
                        uf.merge(cell_idx_1, cell_idx_2);
                    }
                }
            }

            (
                cell_graph,
                uf.leaders_and_sizes()
                    .iter()
                    .map(|x| *x)
                    .filter(|(_, size)| *size >= 2)
                    .collect(),
            )
        }

        pub fn find_way_to_connect_by_specifying_type(
            &self,
            computer_type: usize,
        ) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
            let n = get_n();
            let mut cell_graph = vec![vec![]; n.pow(2)];
            let mut visited = Bitboard::default();
            let mut size_of_groups = vec![];

            for leader_cell_idx in 0..n.pow(2) {
                if self.computer_type(leader_cell_idx.to_coord()) != Some(computer_type) {
                    continue;
                }

                let mut que = VecDeque::from(vec![leader_cell_idx]);
                let mut size_of_group = 0;

                while let Some(curr_cell_idx) = que.pop_front() {
                    let (curr_x, curr_y) = curr_cell_idx.to_coord();

                    for next_x in (curr_x + 1)..n {
                        let next_coord = (next_x, curr_y);

                        if let Some(next_computer_type) = self.computer_type(next_coord) {
                            if next_computer_type == computer_type {
                                let next_cell_idx = next_coord.to_cell_idx();

                                cell_graph[curr_cell_idx].push(next_cell_idx);
                                cell_graph[next_coord.to_cell_idx()].push(curr_cell_idx);

                                if !visited.get(next_coord) {
                                    visited.set(next_coord, true);
                                    que.push_back(next_cell_idx);
                                    size_of_group += 1;
                                }
                            }

                            break;
                        }
                    }

                    for next_y in (curr_y + 1)..n {
                        let next_coord = (curr_x, next_y);

                        if let Some(next_computer_type) = self.computer_type(next_coord) {
                            if next_computer_type == curr_cell_idx {
                                let next_cell_idx = next_coord.to_cell_idx();

                                cell_graph[curr_cell_idx].push(next_cell_idx);
                                cell_graph[next_coord.to_cell_idx()].push(curr_cell_idx);

                                if !visited.get(next_coord) {
                                    visited.set(next_coord, true);
                                    que.push_back(next_cell_idx);
                                    size_of_group += 1;
                                }
                            }

                            break;
                        }
                    }
                }

                size_of_groups.push((leader_cell_idx, size_of_group));
            }

            (cell_graph, size_of_groups)
        }
    }

    impl From<MapByChars> for ServerRoom {
        fn from(init_computer_map: MapByChars) -> Self {
            let (n, k) = (get_n(), get_k());

            let mut bitboards = vec![Bitboard::new(); k];

            for cell_idx in 0..n.pow(2) {
                let (x, y) = cell_idx.to_coord();
                let c = init_computer_map[x][y];
                if c != '0' {
                    let computer_type = c.to_digit(10).unwrap() as usize - 1;
                    bitboards[computer_type].set((x, y), true);
                }
            }

            Self(bitboards)
        }
    }

    pub struct ConnectServerRoom {
        server_room: ServerRoom,
        cable: Bitboard,
        connect_way: Vec<(Coord, Coord)>,
        uf: UnionFind,
    }

    impl From<ServerRoom> for ConnectServerRoom {
        fn from(server_room: ServerRoom) -> Self {
            Self {
                server_room,
                cable: Bitboard::new(),
                connect_way: vec![],
                uf: UnionFind::new(get_n().pow(2)),
            }
        }
    }

    impl ConnectServerRoom {
        pub fn cable_crossing(&self, coord1: Coord, coord2: Coord) -> bool {
            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                let (min_y, max_y) = min_max(y1, y2);
                !self.cable.any_rect(x1..=x2, (min_y + 1)..max_y)
            } else {
                let (min_x, max_x) = min_max(x1, x2);
                !self.cable.any_rect((min_x + 1)..max_x, y1..=y2)
            }
        }

        pub fn connected(&mut self, coord1: Coord, coord2: Coord) -> bool {
            assert_ne!(
                coord1, coord2,
                "The coordinates of `coord1` and `coord2` are the same."
            );

            self.uf.same(coord1.to_cell_idx(), coord2.to_cell_idx())
        }

        pub fn directly_connectable(&mut self, coord1: Coord, coord2: Coord) -> bool {
            if self.connected(coord1, coord2) {
                return false;
            }

            if !self.server_room.directly_connectable(coord1, coord2) {
                return false;
            }

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                let (min_y, max_y) = min_max(y1, y2);
                !self.cable.any_rect(x1..=x2, (min_y + 1)..max_y)
            } else {
                let (min_x, max_x) = min_max(x1, x2);
                !self.cable.any_rect((min_x + 1)..max_x, y1..=y2)
            }
        }

        pub fn connect(&mut self, coord1: Coord, coord2: Coord) {
            // assert!(self.directly_connectable(coord1, coord2));

            let (x1, y1) = coord1;
            let (x2, y2) = coord2;

            if x1 == x2 {
                let (min_y, max_y) = min_max(y1, y2);
                self.cable.set_rect(x1..=x1, (min_y + 1)..max_y, true);
            } else {
                let (min_x, max_x) = min_max(x1, x2);
                self.cable.set_rect((min_x + 1)..max_x, y1..=y2, true);
            }

            self.connect_way.push((coord1, coord2));
            self.uf.merge(coord1.to_cell_idx(), coord2.to_cell_idx());
        }

        pub fn calc_score_for_simple_groups(&mut self) -> usize {
            self.uf.leaders_and_sizes().iter().map(|(_, size)| if *size == 0 {
                0
            } else {
                size * (size - 1) / 2
            }).sum()
        }

        pub fn calc_score(&mut self) -> usize {
            let n = get_n();
            let k = get_k();

            let mut num_each_type_for_each_group = vec![vec![0_usize; k]; n.pow(2)];
            for cell_idx in 0..n.pow(2) {
                if let Some(computer_type) = self.server_room.computer_type(cell_idx.to_coord()) {
                    num_each_type_for_each_group[self.uf.leader(cell_idx)][computer_type] += 1;
                }
            }

            num_each_type_for_each_group
                .into_iter()
                .map(|num_each_type| {
                    let sum: usize = num_each_type.iter().sum();
                    num_each_type
                        .into_iter()
                        .map(|x| {
                            if x == 0 {
                                0
                            } else {
                                x * (x - 1) / 2 - x * (sum - x)
                            }
                        })
                        .sum::<usize>()
                })
                .sum()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StatusNode {
        /// Server room status
        server_room: ServerRoom,

        /// Number of moves
        turn: usize,

        /// Parent node
        parent: Option<usize>,

        /// Previous moves
        prev_moves: Vec<(Coord, Coord)>,

        /// Score
        score: usize,
    }

    impl StatusNode {
        fn construct(server_room: ServerRoom, turn: usize, parent: Option<usize>, prev_moves: Vec<(Coord, Coord)>) -> Self {
            let mut node = Self { server_room, turn, parent, prev_moves, score: 0 };
            node.score = node.connect_way_and_score_when_connect_same_type().1;
            node
        }

        /// Constructs a state node from the initial computer map.
        pub fn from_init_computer_map(init_computer_map: MapByChars) -> Self {
            StatusNode::construct(ServerRoom::from(init_computer_map), 0, None, vec![])
        }

        /// Gets the number of times the computer has been moved.
        pub fn turn(&self) -> usize {
            self.turn
        }

        /// Gets the number of turns remaining.
        pub fn rem_turn(&self) -> usize {
            get_max_turn() - self.turn
        }

        /// Gets the score when the cable is connected in the prescribed rule.
        pub fn score(&self) -> usize {
            self.score
        }

        /// Shows the status of the server room.
        pub fn show_server_room_with_config(&self, one_based: bool, empty_symbol: char) {
            self.server_room.show_with_config(one_based, empty_symbol);
        }

        /// Shows the status of the server room.
        pub fn show_server_room(&self) {
            self.server_room.show();
        }

        /// Returns the node after the computer has been moved once.
        pub fn one_moved(&self, node_idx: usize, coord: Coord, dir: Dir) -> Self {
            assert_ne!(self.rem_turn(), 0, "No more computers can be moved.");

            let mut moved_server_room = self.server_room.clone();
            moved_server_room.move_one(coord, dir);

            StatusNode::construct(moved_server_room, self.turn() + 1, Some(node_idx), vec![(coord, coord.moved(dir))])
        }

        pub fn moved(&self, node_idx: usize, source_coord: Coord, move_dirs: &[Dir]) -> Self {
            let mut coord = source_coord;
            let mut moved_server_room = self.server_room.clone();
            let mut prev_moves = vec![];
            
            for &dir in move_dirs.iter() {
                moved_server_room.move_one(coord, dir);
                let moved_coord = coord.moved(dir);
                prev_moves.push((coord, moved_coord));
                coord = moved_coord;
            }

            Self::construct(moved_server_room, self.turn() + move_dirs.len(), Some(node_idx), prev_moves)
        }

        pub fn connect_same_type_group(
            &self,
            connect_server_room: &mut ConnectServerRoom,
            cell_graph: &Vec<Vec<usize>>,
            leader_cell_idx: usize,
            rem_turn: &mut usize,
        ) {
            if *rem_turn == 0 {
                return;
            }

            let mut que = VecDeque::from(vec![leader_cell_idx]);

            while let Some(source_cell_idx) = que.pop_front() {
                let source_coord = source_cell_idx.to_coord();

                for &dest_cell_idx in cell_graph[source_cell_idx].iter() {
                    let dest_coord = dest_cell_idx.to_coord();

                    assert_ne!(source_coord, dest_coord, "Cannot connect to itself.");

                    let (source_x, source_y) = source_coord;
                    let (dest_x, dest_y) = dest_coord;

                    assert!(
                        source_x == dest_x || source_y == dest_y,
                        "The computers to be connected must have matching rows and columns."
                    );

                    if connect_server_room.directly_connectable(source_coord, dest_coord) {
                        connect_server_room.connect(source_coord, dest_coord);
                        *rem_turn -= 1;
                        que.push_back(dest_cell_idx);

                        if *rem_turn == 0 {
                            return;
                        }
                    }
                }
            }
        }

        pub fn connect_way_and_score_when_connect_one_type(&self) -> (Vec<(Coord, Coord)>, usize) {
            let mut rem_turn = self.rem_turn();
            if rem_turn == 0 {
                return (vec![], 0);
            }

            let mut max_score = 0;
            let mut connect_way = vec![];

            for computer_type in 0..get_k() {
                let mut connect_server_room = ConnectServerRoom::from(self.server_room.clone());
                let (cell_graph, mut leaders_and_sizes) = connect_server_room
                    .server_room
                    .find_way_to_connect_by_specifying_type(computer_type);

                leaders_and_sizes.sort_by_key(|x| Reverse(x.1));
                leaders_and_sizes
                    .into_iter()
                    .for_each(|(leader_cell_idx, _)| {
                        self.connect_same_type_group(
                            &mut connect_server_room,
                            &cell_graph,
                            leader_cell_idx,
                            &mut rem_turn,
                        )
                    });

                let score = connect_server_room.calc_score_for_simple_groups();

                max_score = max_score.max(score);
                connect_way = connect_server_room.connect_way;
            }

            (connect_way, max_score)
        }

        pub fn connect_way_and_score_when_connect_same_type(&self) -> (Vec<(Coord, Coord)>, usize) {
            let mut rem_turn = self.rem_turn();
            if rem_turn == 0 {
                return (vec![], 0);
            }

            let mut connect_server_room = ConnectServerRoom::from(self.server_room.clone());
            let (cell_graph, mut leaders_and_sizes) = connect_server_room
                .server_room
                .find_way_to_connect_same_type();

            leaders_and_sizes.sort_by_key(|x| Reverse(x.1));
            leaders_and_sizes
                .into_iter()
                .for_each(|(leader_cell_idx, _)| {
                    self.connect_same_type_group(
                        &mut connect_server_room,
                        &cell_graph,
                        leader_cell_idx,
                        &mut rem_turn,
                    )
                });

            let score = connect_server_room.calc_score_for_simple_groups();
            (connect_server_room.connect_way, score)
        }

        pub fn show_connect_way(&self) {
            let connect_way = self.connect_way_and_score_when_connect_same_type().0;

            println!("{}", connect_way.len());
            for ((x1, y1), (x2, y2)) in connect_way {
                println!("{} {} {} {}", x1, y1, x2, y2);
            }
        }

        pub fn random_select_movable_computer<R>(&self, rng: &mut R) -> (Coord, Vec<Dir>)
        where
            R: Rng,
        {
            self.server_room.random_select_movable_computer(rng)
        }
    }

    #[derive(Debug)]
    pub struct StatusTree(Vec<StatusNode>);

    impl StatusTree {
        pub fn new(init_computer_map: MapByChars) -> Self {
            Self(vec![StatusNode::from_init_computer_map(init_computer_map)])
        }

        pub fn node_num(&self) -> usize {
            self.0.len()
        }

        pub fn get_node(&self, node_idx: usize) -> &StatusNode {
            &self.0[node_idx]
        }

        fn get_node_mut(&mut self, node_idx: usize) -> &mut StatusNode {
            &mut self.0[node_idx]
        }

        pub fn add_moved_node(&mut self, node_idx: usize, source_coord: Coord, move_dirs: &[Dir]) -> usize {
            self.0.push(self.get_node(node_idx).moved(node_idx, source_coord, move_dirs));
            self.node_num() - 1
        }

        pub fn add_random_moved_node<R>(&mut self, node_idx: usize, max_dist: usize, rng: &mut R) -> usize
        where
            R: Rng,
        {
            let node = self.get_node(node_idx);
            let source_coord = node.server_room.random_select_movable_computer(rng).0;
            let move_dirs = node.server_room.random_move_route(source_coord, max_dist, rng).unwrap();
            self.add_moved_node(node_idx, source_coord, &move_dirs)
        }

        pub fn add_one_moved_node(&mut self, node_idx: usize, coord: Coord, dir: Dir) -> usize {
            let moved_node = self.get_node_mut(node_idx).one_moved(node_idx, coord, dir);

            self.0.push(moved_node);

            self.node_num() - 1
        }

        pub fn move_history(&self, node: &StatusNode) -> Vec<(Coord, Coord)> {
            let mut curr_node = node;
            let mut prev_moves = curr_node.prev_moves.clone();
            prev_moves.reverse();
            let mut move_history: Vec<(Coord, Coord)> = prev_moves;

            while let Some(prev_node_idx) = curr_node.parent
            {
                curr_node = self.get_node(prev_node_idx);
                curr_node.prev_moves.iter().rev().for_each(|(source_coord, dest_coord)| move_history.push((*source_coord, *dest_coord)));
                // move_history.push(coord_pair);
                curr_node = self.get_node(prev_node_idx);
            }

            move_history.reverse();

            move_history
        }

        pub fn show_move_history(&self, node: &StatusNode) {
            let move_history = self.move_history(node);

            println!("{}", move_history.len());
            for ((source_x, source_y), (dest_x, dest_y)) in move_history {
                println!("{} {} {} {}", source_x, source_y, dest_x, dest_y);
            }
        }

        pub fn find_best_node(&self) -> &StatusNode {
            self.0.iter().max_by_key(|node| node.score()).unwrap()
        }

        pub fn show_node_solution(&self, node: &StatusNode) {
            self.show_move_history(node);
            node.show_connect_way();
        }
    }
}

pub mod atcoder8_library {
    pub mod direction {
        /// Enumerator for direction
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Dir {
            /// Represents the up.
            Up,

            /// Represents the down.
            Down,

            /// Represents the left.
            Left,

            /// Represents the right.
            Right,
        }

        impl Dir {
            /// Creates an array of all directions.
            ///
            /// Returns `vec![Dir::Up, Dir::Down, Dir::Left, Dir::Right]`.
            pub fn all_directions() -> Vec<Dir> {
                vec![Dir::Up, Dir::Down, Dir::Left, Dir::Right]
            }

            /// Gets the opposite direction.
            ///
            /// The return values corresponding to the arguments are as follows:
            ///
            /// * `Dir::Up` => `Dir::Down`
            /// * `Dir::Down` => `Dir::Up`
            /// * `Dir::Left` => `Dir::Right`
            /// * `Dir::Right` => `Dir::Left`
            pub fn opposite(self) -> Self {
                match self {
                    Self::Up => Dir::Down,
                    Self::Down => Dir::Up,
                    Self::Left => Dir::Right,
                    Self::Right => Dir::Left,
                }
            }

            /// Converts `Dir` to `usize`.
            ///
            /// The conversion rules are as follows:
            ///
            /// * `Dir::Up` => `0`
            /// * `Dir::Down` => `1`
            /// * `Dir::Left` => `2`
            /// * `Dir::Right` => `3`
            pub fn to_usize(self) -> usize {
                match self {
                    Dir::Up => 0,
                    Dir::Down => 1,
                    Dir::Left => 2,
                    Dir::Right => 3,
                }
            }

            /// Converts `Dir` to `char`.
            ///
            /// The conversion rules are as follows:
            ///
            /// * `Dir::Up` => `'U'`
            /// * `Dir::Down` => `'D'`
            /// * `Dir::Left` => `'L'`
            /// * `Dir::Right` => `'R'`
            pub fn to_char(self) -> char {
                match self {
                    Dir::Up => 'U',
                    Dir::Down => 'D',
                    Dir::Left => 'L',
                    Dir::Right => 'R',
                }
            }

            /// Converts `Dir` to `&'static str`
            ///
            /// The conversion rules are as follows:
            ///
            /// * `Dir::Up` => `"Up"`
            /// * `Dir::Down` => `"Down"`
            /// * `Dir::Left` => `"Left"`
            /// * `Dir::Right` => `"Right"`
            pub fn to_str(self) -> &'static str {
                match self {
                    Dir::Up => "Up",
                    Dir::Down => "Down",
                    Dir::Left => "Left",
                    Dir::Right => "Right",
                }
            }
        }

        impl From<usize> for Dir {
            fn from(dir_idx: usize) -> Self {
                match dir_idx {
                    0 => Dir::Up,
                    1 => Dir::Down,
                    2 => Dir::Left,
                    3 => Dir::Right,
                    _ => panic!("An undefined direction was specified."),
                }
            }
        }
    }

    pub mod union_find {
        //! Union-Find processes the following queries for an edgeless graph in `O(α(n))` amortized time.
        //! * Add an undirected edge.
        //! * Deciding whether given two vertices are in the same connected component
        //!
        //! When a method is called, route compression is performed as appropriate.

        /// Union-Find processes the following queries for an edgeless graph in `O(α(n))` amortized time.
        /// * Add an undirected edge.
        /// * Deciding whether given two vertices are in the same connected component
        ///
        /// When a method is called, route compression is performed as appropriate.
        ///
        /// # Examples
        ///
        /// ```
        /// use union_find::UnionFind;
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

            /// If it is not representative, the index of its parent is stored.
            group_num: usize,
        }

        impl UnionFind {
            /// Constructs an undirected graph with `n` vertices and 0 edges.
            pub fn new(n: usize) -> Self {
                UnionFind {
                    parent_or_size: vec![-1; n],
                    group_num: n,
                }
            }

            /// Returns the representative of the connected component that contains the vertex `a`.
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

                // Returns a representative of the connected component.
                curr
            }

            /// Checks if vertices `a` and `b` are in the same connected component.
            pub fn same(&mut self, a: usize, b: usize) -> bool {
                self.leader(a) == self.leader(b)
            }

            /// Adds an edge between vertex `a` and vertex `b`.
            /// Returns true if the connected component to which `a` belongs and that of `b` are newly combined.
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

                // Returns true because the connected component is newly combined.
                true
            }

            /// Returns the size of the connected component that contains the vertex `a`.
            pub fn size(&mut self, a: usize) -> usize {
                let leader = self.leader(a);
                -self.parent_or_size[leader] as usize
            }

            /// Adds a new vertex with degree zero.
            pub fn add(&mut self) {
                self.parent_or_size.push(-1);
                self.group_num += 1;
            }

            /// Returns the number of connected components.
            pub fn group_num(&self) -> usize {
                self.group_num
            }

            /// Returns the number of elements.
            pub fn elem_num(&self) -> usize {
                self.parent_or_size.len()
            }

            /// Returns a list of connected components.
            pub fn groups(&mut self) -> Vec<Vec<usize>> {
                let n = self.parent_or_size.len();
                let mut groups = vec![vec![]; n];

                for i in 0..n {
                    groups[self.leader(i)].push(i);
                }

                groups.retain(|x| !x.is_empty());

                groups
            }

            /// Returns a list of leaders of each connected component.
            pub fn leaders(&mut self) -> Vec<usize> {
                (0..self.parent_or_size.len())
                    .filter(|&i| self.leader(i) == i)
                    .collect()
            }

            /// Creates a list of leaders and sizes for each connected component.
            pub fn leaders_and_sizes(&mut self) -> Vec<(usize, usize)> {
                self.leaders()
                    .iter()
                    .map(|&leader| (leader, self.size(leader)))
                    .collect()
            }
        }
    }
}
