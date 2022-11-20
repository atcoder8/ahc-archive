#!/bin/bash

cargo build --release --package submission &&
    cargo run --package evaluation --release \
        -- -c ./evaluation/config_2000.toml \
        -o ./evaluation/result_2000.csv
