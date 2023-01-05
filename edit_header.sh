#!/bin/bash

while IFS= read -r line; do

  awk 'NR>=50' "$line"/hlist_1.00000.list > "$line"/hlist.list

done < "halo_names.txt"