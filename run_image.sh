#!/bin/bash
media_path="@thirdparty/image.jpg"
cog predict -i "media_path=$media_path" | \
tee \
>(jq -r '.debug_media | split(",")[1]' | base64 -d > debug_output.png) \
>(jq -r '.mediapipe' > mediapipe.json) \
>(jq -r '.fullbody' > fullbody.json) \
>(jq -r '.hand_landmarks' > hand_landmarks.json) \
>(jq -r '.blendshapes' > blendshapes.json) \
>/dev/null
