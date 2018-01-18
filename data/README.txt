bottleneck features extrated from resnet without top layer in keras

bottleneck_features_face_cropped.npy: size (11338,1,1,2048) The face is cropped out, aligned and further resize to 224 x 224. If a face is not detected, the original picture is resize to 224 x 224 preserving width/height ratio. The empty space is padded with border color calculated from the mean value a random patch in the picture.

bottleneck_features.npy: size (11338,1,1,2048). No face cropping. original picture directly resized to 224 x 224 with black boarder padding.

labels.npy: label vector (11338,1)

