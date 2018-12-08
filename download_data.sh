#!/bin/bash


# chech what is it
# -e  Exit immediately if a command exits with a non-zero status.
# -x  Print commands and their arguments as they are executed.

set -ex

# for label-maker


for i in 4979 4968 4961 4960 4959 4954 4950 4945 4943 4942 4940 4937 # list of projects to consider
do
	# corresponding config files for zoom 17 and 18
	name1="zoom17/config_$i.json"
	save1="data/17_$i"
	name2="zoom18/config_$i.json"
	save2="data/18_$i"
	
	# download and save for zoom 17
    label-maker download -c $name1 -d $save1
	label-maker labels -c $name1 -d $save1
	label-maker images -c $name1 -d $save1
	
	# remove everything except two folders: tiles and labels
	mv $save1/labels data/temp/labels
	mv $save1/tiles data/temp/tiles

	rm -rf $save1/*

	mv data/temp/labels $save1/labels
	mv data/temp/tiles $save1/tiles 
	
	# download and save for zoom 18
	label-maker download -c $name2 -d $save2
	label-maker labels -c $name2 -d $save2
	label-maker images -c $name2 -d $save2
	
	# remove everything except two folders: tiles and labels
	mv $save2/labels data/temp/labels
	mv $save2/tiles data/temp/tiles

	rm -rf $save2/*

	mv data/temp/labels $save2/labels
	mv data/temp/tiles $save2/tiles 
done

