#!/bin/bash
# stab.sh src_dir dst_dir [clips...]  -> vidstab-smoothed copies (same fps/size, auto zoom crop)
src=$1; dst=$2; shift 2; mkdir -p $dst
clips="$@"; [ -z "$clips" ] && clips=$(cd $src; ls *.MOV)
for c in $clips; do
  t=/tmp/stab_$$_$c.trf
  ffmpeg -v error -y -i $src/$c -vf vidstabdetect=shakiness=7:accuracy=15:result=$t -f null -
  ffmpeg -v error -y -i $src/$c -vf vidstabtransform=input=$t:smoothing=${SMOOTH:-60}:optzoom=1:zoomspeed=0.1:interpol=bicubic,unsharp=5:5:0.3 \
     -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p -an $dst/$c
  rm -f $t; echo "stab $c"
done
