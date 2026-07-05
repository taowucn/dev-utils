#!/bin/bash
set -e

out_dir=$1

if [ -z "${out_dir}" ]; then
    echo "Usage: $0 <out_dir>"
    exit 1
fi 
echo "Install to ${out_dir}"

for src_dir in `ls -d */`; do
    src_dir=${src_dir%/}

    echo "cp ${src_dir}/* to ${out_dir}"
    cp -r ${src_dir}/* ${out_dir}/
    chmod +x ${out_dir}/*
done
