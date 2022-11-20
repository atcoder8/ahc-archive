#!/bin/bash

readonly SUBMISSION_FILE="./submission/src/main.rs"
readonly RECORD_DIR="./submission/src/bin"
readonly RECORD_NUMBER=$(printf "%04d" $(ls ${RECORD_DIR} | wc -l))
readonly RECORD_FILE="./submission/src/bin/${RECORD_NUMBER}.rs"

echo "\

[[bin]]
name = \"submission-${RECORD_NUMBER}\"
path = \"src/bin/${RECORD_NUMBER}.rs\"\
" >> "./submission/Cargo.toml"

if [ ! -d ${RECORD_DIR} ]; then
  mkdir -p ${RECORD_DIR}
  echo "Create directory \`${RECORD_DIR}\`."
fi

cp ${SUBMISSION_FILE} ${RECORD_FILE}

echo "Copied \`${SUBMISSION_FILE}\` to \`${RECORD_FILE}\`."
