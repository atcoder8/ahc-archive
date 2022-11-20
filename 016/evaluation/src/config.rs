use serde::Deserialize;

/// Paths to executable files.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutableFiles {
    /// Path to the executable file of the code for submission.
    #[serde(default = "ExecutableFiles::default_submission")]
    pub submission: String,

    /// Path to the executable file of the tester.
    #[serde(default = "ExecutableFiles::default_tester")]
    pub tester: String,
}

impl ExecutableFiles {
    pub const DEFAULT_SUBMISSION: &'static str = "./target/release/submission";

    pub const DEFAULT_TESTER: &'static str = "./target/release/tester";

    fn default_submission() -> String {
        String::from(ExecutableFiles::DEFAULT_SUBMISSION)
    }

    fn default_tester() -> String {
        String::from(ExecutableFiles::DEFAULT_TESTER)
    }
}

impl Default for ExecutableFiles {
    fn default() -> Self {
        Self {
            submission: ExecutableFiles::default_submission(),
            tester: ExecutableFiles::default_tester(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Paths {
    /// Paths to the executable files.
    #[serde(default = "ExecutableFiles::default")]
    pub executable_files: ExecutableFiles,

    /// Path to the file containing the seed values.
    #[serde(default = "Paths::default_seeds_file")]
    pub seeds_file: String,

    /// Path to the input directory.
    #[serde(default = "Paths::default_input_dir")]
    pub input_dir: String,

    /// Path to the output directory.
    #[serde(default = "Paths::default_output_dir")]
    pub output_dir: String,
}

impl Paths {
    pub const DEFAULT_SEEDS_FILE: &'static str = "./tools/seeds.txt";

    pub const DEFAULT_INPUT_DIR: &'static str = "./tools/in";

    pub const DEFAULT_OUTPUT_DIR: &'static str = "./evaluation/out";

    fn default_seeds_file() -> String {
        String::from(Paths::DEFAULT_SEEDS_FILE)
    }

    fn default_input_dir() -> String {
        String::from(Paths::DEFAULT_INPUT_DIR)
    }

    fn default_output_dir() -> String {
        String::from(Paths::DEFAULT_OUTPUT_DIR)
    }
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            executable_files: ExecutableFiles::default(),
            seeds_file: Paths::default_seeds_file(),
            input_dir: Paths::default_input_dir(),
            output_dir: Paths::default_output_dir(),
        }
    }
}

/// Settings for parallel processing.
#[derive(Debug, Clone, Deserialize)]
pub struct Thread {
    /// Number of thread used to execute the code for submission.
    #[serde(default = "Thread::default_thread_num")]
    pub thread_num: usize,
}

impl Thread {
    pub const DEFAULT_THREAD_NUM: usize = 8;

    fn default_thread_num() -> usize {
        Thread::DEFAULT_THREAD_NUM
    }
}

impl Default for Thread {
    fn default() -> Self {
        Self {
            thread_num: Thread::default_thread_num(),
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct Config {
    /// Paths to executable files.
    #[serde(default = "Paths::default")]
    pub paths: Paths,

    /// Settings for parallel processing.
    #[serde(default = "Thread::default")]
    pub thread: Thread,
}
