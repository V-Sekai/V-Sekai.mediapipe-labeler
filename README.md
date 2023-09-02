# Mediapipe Blendshape Labeler

**WORK IN PROGRESS**

<img src="image.jpg_debug.jpg" width="25%">

[Blendshapes JSON](mediapipe_blendshape_labeler/image.jpg_blendshapes.json)

# Install

```zsh
brew install cog
# Optional
#wget -O thirdparty/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
cog predict -i image=@./thirdparty/image.jpg
```
