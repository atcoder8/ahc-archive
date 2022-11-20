#!/bin/bash

cargo build --release --package submission &&
    cargo run --package evaluation --release \
        -- -c ./evaluation/config_100.toml \
        -o ./evaluation/result_100.csv
