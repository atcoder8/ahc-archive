#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: new_problem [Submission Number]"
    exit 1
fi

if [ -f src/bin/$1.rs ]; then
    echo "\"src/bin/$1.rs\" is already exists."
    exit 1
fi

echo "
[[bin]]
name = \"submission-$1\"
path = \"src/bin/$1.rs\"" \
    >> ./submission/Cargo.toml

echo "\
fn main() {
    
}" \
    >>./submission/src/bin/$1.rs
