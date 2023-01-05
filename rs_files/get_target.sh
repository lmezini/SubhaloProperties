#!/bin/bash

while IFS= read -r line; do

  wget https://www.slac.stanford.edu/~yymao/mw_PLAFAO5I69P4L8I4/"$line"/target_halo.txt
  mv target_halo.txt "$line"/
done < "halo_names.txt"
