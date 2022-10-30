#!/bin/bash

cargo build --release --package submission && cargo run --package evaluation --release -- -c ./evaluation/100_config.toml -o ./evaluation/100_evaluation_result.csv
