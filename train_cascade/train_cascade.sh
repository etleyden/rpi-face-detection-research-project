#!/bin/zsh

# compile the positive and negative samples into opencv friendly files
# AVAILABLE POS SAMPLES: 13386
NUM_POS_SAMPLES=-1

#rm positives.info
#python train_cascade/train_cascade.py 


# try different size windows?
# select 1317 samples for a 1:1 of positive to negative samples
# vec file must contain: 
# >= (numPos + (numStages-1) * (1 - minHitRate) * numPos) + S
mkdir classifier

echo "Creating Samples\n"

rm positives.vec

opencv_createsamples -info positives.info -vec positives.vec -num 1000 -w 24 -h 24

echo "Train Cascade\n"

opencv_traincascade -data classifier -vec positives.vec -bg negatives.txt -numPos 1000 -numNeg 1317 -numStages 10 -w 24 -h 24
