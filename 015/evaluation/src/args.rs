use clap::Parser;

/// If no output results file is specified by the command line arguments,
/// this pathname is used.
const DEFAULT_OUTPUT_RESULTS_FILE_PATHNAME: &str = "evaluation_result.csv";

#[derive(Debug, Parser)]
#[clap(author, version)]
pub struct Args {
    /// Pathname of the Configuration file.
    ///
    /// If no configuration file is specified, default values are used.
    #[clap(short = 'c', long = "config")]
    pub config_file_pathname: Option<String>,

    /// Pathname of the file to output evaluation results
    #[clap(short = 'o', long = "output_results", default_value_t = String::from(DEFAULT_OUTPUT_RESULTS_FILE_PATHNAME))]
    pub output_results_file_pathname: String,
}
