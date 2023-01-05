#!/bin/bash                                                                           
while IFS= read -r line; do

  rm /Users/lmezini/proj_2/Halos_Recalculated/"$line"/out_2.txt
  /Users/lmezini/proj_2/rockstar/util/find_parents /Users/lmezini/proj_2/Halos_Recalculated/"$line"/out_0.list 125 > /Users/lmezini/proj_2/Halos_Recalculated/"$line"/out_2.txt

done < "halo_names.txt"