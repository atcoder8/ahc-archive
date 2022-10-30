#!/bin/bash

cargo build --release --package submission && cargo run --package evaluation --release -- -c ./evaluation/2000_config.toml -o ./evaluation/2000_evaluation_result.csv
