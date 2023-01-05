#!/bin/bash

while IFS= read -r line; do

  wget https://www.slac.stanford.edu/~yymao/mw_PLAFAO5I69P4L8I4/"$line"/hlists/hlist_1.00000.list
  mv hlist_1.00000.list "$line"/
done < "halo_names.txt"
