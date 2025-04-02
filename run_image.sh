media_path="@thirdparty/image.jpg"
cog predict -i "media_path=$media_path" | \
tee \
>(jq -r '.debug_media | split(",")[1]' | base64 -d > debug_output.png) \
>(jq -r '.coco_keypoints' > coco_keypoints.json) \
>(jq -r '.blendshapes' > blendshapes.json) \
>/dev/null
