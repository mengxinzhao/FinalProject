#!/bin/sh
# convert to png to use less storage space on colorferet dataset
#
for dir in `ls`
do  
	chmod +w $dir
	echo $dir
        cd $dir
        bunzip2 *.bz2 
        for file in `ls *.ppm`
        do
		 name=${file%.ppm}.png
		 if [ ! -f $name ]; then
		 	ffmpeg -i $file $name
		 fi
	done
	rm -f *.bz2
        rm -f *.ppm
	cd - 
done
