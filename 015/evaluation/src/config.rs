use serde::Deserialize;

/// Paths to executable files.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutableFiles {
    /// Path to the executable file of the code for submission.
    #[serde(default = "ExecutableFiles::default_submission")]
    pub submission: String,

    /// Path to the executable file of the visualizer.
    #[serde(default = "ExecutableFiles::default_vis")]
    pub vis: String,
}

impl ExecutableFiles {
    pub const DEFAULT_SUBMISSION: &'static str = "./target/release/submission";

    pub const DEFAULT_VIS: &'static str = "./target/release/vis";

    fn default_submission() -> String {
        String::from(ExecutableFiles::DEFAULT_SUBMISSION)
    }

    fn default_vis() -> String {
        String::from(ExecutableFiles::DEFAULT_VIS)
    }
}

impl Default for ExecutableFiles {
    fn default() -> Self {
        Self {
            submission: ExecutableFiles::default_submission(),
            vis: ExecutableFiles::default_vis(),
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

    /// Path to the input folder.
    #[serde(default = "Paths::default_input_folder")]
    pub input_folder: String,

    /// Path to the output folder.
    #[serde(default = "Paths::default_output_folder")]
    pub output_folder: String,

    /// Path to the folder for output of visualized images.
    #[serde(default = "Paths::default_vis_folder")]
    pub vis_folder: String,
}

impl Paths {
    pub const DEFAULT_SEEDS_FILE: &'static str = "./tools/seeds.txt";

    pub const DEFAULT_INPUT_FOLDER: &'static str = "./tools/in";

    pub const DEFAULT_OUTPUT_FOLDER: &'static str = "./evaluation/out";

    pub const DEFAULT_VIS_FOLDER: &'static str = "./evaluation/vis";

    fn default_seeds_file() -> String {
        String::from(Paths::DEFAULT_SEEDS_FILE)
    }

    fn default_input_folder() -> String {
        String::from(Paths::DEFAULT_INPUT_FOLDER)
    }

    fn default_output_folder() -> String {
        String::from(Paths::DEFAULT_OUTPUT_FOLDER)
    }

    fn default_vis_folder() -> String {
        String::from(Paths::DEFAULT_VIS_FOLDER)
    }
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            executable_files: ExecutableFiles::default(),
            seeds_file: Paths::default_seeds_file(),
            input_folder: Paths::default_input_folder(),
            output_folder: Paths::default_output_folder(),
            vis_folder: Paths::default_vis_folder(),
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
