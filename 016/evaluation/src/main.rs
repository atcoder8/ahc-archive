use std::{
    fs::{create_dir_all, read_to_string, File},
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use clap::Parser;
use evaluation::{args::Args, config::Config, time_measurement::StopWatch};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::izip;
use rayon::{
    prelude::{IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};

fn main() {
    // Reads command line arguments.
    let args = Args::parse();

    // Reads the configuration.
    let config = if let Some(config_file_pathname) = &args.config_file_pathname {
        let config =
            read_to_string(config_file_pathname).expect("Failed to load configuration file.");
        toml::from_str(&config).expect("Failed to deserialize.")
    } else {
        Config::default()
    };

    ThreadPoolBuilder::new()
        .num_threads(config.thread.thread_num)
        .build_global()
        .expect("Failed to initialize global thread pool.");

    // Reads the list of seeds.
    let seeds: Vec<usize> = read_to_string(&config.paths.seeds_file)
        .expect("Failed to load seed.")
        .split_whitespace()
        .map(|x| x.parse().expect("Could not read seed as `usize`."))
        .collect();

    let exe_num = seeds.len();

    assert_ne!(exe_num, 0, "Seed is not specified.");

    // Creates output directory.
    create_dir_all(Path::new(&config.paths.output_dir))
        .expect("Failed to create output directory.");

    // Style of progress bar
    let progress_style = ProgressStyle::template(
        ProgressStyle::default_bar(),
        "{prefix}\n{wide_bar} {pos:>3}/{len:3} {percent:>3}% [{elapsed_precise}]",
    )
    .unwrap();

    // Uses the progress bar to display the progress of the program execution for test.
    let progress_bar = ProgressBar::new(exe_num as u64);
    progress_bar.set_style(progress_style);
    progress_bar.set_prefix("[Tester] Running...");

    // Execute test and get a score.
    let (scores, exe_times) = seeds
        .par_iter()
        .map(|&seed| {
            let score = run_tester(&config, seed);
            progress_bar.inc(1);
            score
        })
        .unzip();

    // Clears the progress bar.
    progress_bar.finish_and_clear();

    // Outputs the number of executions.
    println!("Number of executions: {}", exe_num);

    println!();

    // Shows score statistics.
    show_score_statistics(&seeds, &scores);

    println!();

    // Show execution time statistics.
    show_exe_time_statistics(&seeds, &exe_times);

    // Creates output results directory.
    create_dir_all(
        Path::new(&args.output_results_file_pathname)
            .parent()
            .unwrap(),
    )
    .expect("Failed to create vis directory.");

    // Opens a file to output evaluation results.
    let mut output_results_file =
        File::create(&args.output_results_file_pathname).expect("Failed to create output file.");

    // Output item names
    output_results_file
        .write_all(b"seed\tscore\texe_time\n")
        .expect("Failed to write to output file.");

    // Output seed, score and execution time as tab delimiter to file.
    for (&seed, &score, &exe_time) in izip!(&seeds, &scores, &exe_times) {
        output_results_file
            .write_all(format!("{}\t{}\t{}\n", seed, score, exe_time).as_bytes())
            .expect("Failed to write to output file.");
    }
}

/// Executes the program for visualization and returns the score.
fn run_tester(config: &Config, seed: usize) -> (usize, f64) {
    let input_file_path = Path::new(&config.paths.input_dir).join(&format!("{:04}.txt", seed));
    let output_file_path = Path::new(&config.paths.output_dir).join(&format!("{:04}.txt", seed));

    let input_text = read_to_string(&input_file_path).expect("Failed to load file.");

    let tester_process = Command::new(&config.paths.executable_files.tester)
        .args([&config.paths.executable_files.submission])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to create process to execute file for visualize.");

    let stop_watch = StopWatch::new();

    tester_process
        .stdin
        .as_ref()
        .unwrap()
        .write_all(input_text.as_bytes())
        .expect("Failed to input into the program execution process for tester.");

    let exe_time = stop_watch.elapsed_time();

    let output = tester_process
        .wait_with_output()
        .expect("Failed to create process to execute file for tester.");

    assert!(
        output.status.success(),
        "
Runtime error occurred (Exit Status: {}).
------------------------------ Error Message ------------------------------
{}
---------------------------------------------------------------------------
",
        output.status,
        String::from_utf8(output.stderr).unwrap(),
    );

    let mut output_file = File::create(&output_file_path).expect("Failed to create output file.");
    output_file
        .write_all(&output.stdout)
        .expect("Failed to write to output file.");

    let stderr = String::from_utf8(output.stderr).unwrap();
    let score = stderr
        .split_whitespace()
        .last()
        .expect("Failed to retrieve score.");
    let score = score.parse().expect("Cannot convert score to `usize`");

    (score, exe_time)
}

/// Shows score statistics.
fn show_score_statistics(seeds: &Vec<usize>, scores: &Vec<usize>) {
    assert_eq!(seeds.len(), scores.len());

    let exe_num = seeds.len();

    let total_score: usize = scores.iter().sum();
    let avg_score = total_score as f64 / exe_num as f64;

    let min_score = *scores.iter().min().unwrap();
    let min_score_idx = scores.iter().position(|&score| score == min_score).unwrap();

    let max_score = *scores.iter().max().unwrap();
    let max_score_idx = scores.iter().position(|&score| score == max_score).unwrap();

    println!(
        "\
[Score Statistics]
Total: {}
Average: {:.3}
Min: {} (seed = {})
Max: {} (seed = {})",
        total_score, avg_score, min_score, seeds[min_score_idx], max_score, seeds[max_score_idx],
    );
}

/// Shows execution time statistics.
fn show_exe_time_statistics(seeds: &Vec<usize>, exe_times: &Vec<f64>) {
    assert_eq!(seeds.len(), exe_times.len());

    let exe_num = seeds.len();

    let total_exe_time: f64 = exe_times.iter().sum();
    let avg_exe_time = total_exe_time / exe_num as f64;

    let min_exe_time = *exe_times
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let min_exe_time_idx = exe_times
        .iter()
        .position(|&exe_time| exe_time == min_exe_time)
        .unwrap();

    let max_exe_time = *exe_times
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_exe_time_idx = exe_times
        .iter()
        .position(|&exe_time| exe_time == max_exe_time)
        .unwrap();

    println!(
        "\
[Execution Time]
Total: {:.3}
Average: {:.3}
Min: {:.3} (seed = {})
Max: {:.3} (seed = {})",
        total_exe_time,
        avg_exe_time,
        min_exe_time,
        seeds[min_exe_time_idx],
        max_exe_time,
        seeds[max_exe_time_idx],
    );
}
