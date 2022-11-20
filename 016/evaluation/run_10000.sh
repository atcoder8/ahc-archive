#!/bin/bash

cargo build --release --package submission --bin submission &&
    cargo run --package evaluation --release \
        -- -c ./evaluation/config_10000.toml \
        -o ./evaluation/result_10000.csv
