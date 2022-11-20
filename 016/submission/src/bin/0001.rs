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

    let noise_rate: usize = epsilon.split('.').skip(1).next().unwrap().parse().unwrap();

    if noise_rate <= 10 {
        strategy::without_noise(graph_num);
    } else {
        strategy::simple(graph_num);
    }
}

pub mod strategy {
    pub fn simple(graph_num: usize) {
        println!("4");
        for _ in 0..graph_num {
            println!("000000");
        }

        for _ in 0..100 {
            let _hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            println!("0");
        }
    }

    pub fn without_noise(graph_num: usize) {
        let n = (4..).find(|&i| i * (i - 1) / 2 + 1 >= graph_num).unwrap();
        let max_edge_num = n * (n - 1) / 2;

        println!("{}", n);
        for i in 0..graph_num {
            println!(
                "{}{}",
                String::from("1").repeat(i),
                String::from("0").repeat(max_edge_num - i)
            );
        }

        for _ in 0..100 {
            let hh: Vec<char> = {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().chars().collect()
            };

            let edge_num = hh.iter().filter(|&&h| h == '1').count();

            let pred = if edge_num < graph_num {
                edge_num
            } else {
                0
            };

            println!("{}", pred);
        }
    }
}
