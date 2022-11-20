fn main() {
    let (m, epsilon) = {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        (
            iter.next().unwrap().parse::<usize>().unwrap(),
            iter.next().unwrap().parse::<String>().unwrap(),
        )
    };

    let noise_rate: usize = epsilon.split('.').skip(1).next().unwrap().parse().unwrap();

    if noise_rate == 0 {
        strategy::noiseless(m);
    } else {
        strategy::simple(m);
    }

    println!("4");
    for _ in 0..m {
        println!("000000");
    }
}

pub mod strategy {
    pub fn simple(m: usize) {
        println!("4");
        for _ in 0..m {
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

    pub fn noiseless(m: usize) {
        let n = (4..).find(|&i| i * (i - 1) / 2 + 1 >= m).unwrap();
        let max_edge_num = n * (n - 1) / 2;

        println!("{}", n);
        for i in 0..m {
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
            println!("{}", edge_num);
        }
    }
}
