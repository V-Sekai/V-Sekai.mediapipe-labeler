media_path="@【まなこ×やっこ】CALL ME CALL ME　踊ってみた【オリジナル振付】 [75izBsrw-sw].mp4"
cog predict -i test_mode=True -i "media_path=$media_path" | \
tee \
>(jq -r '.debug_media | split(",")[1]' | base64 -d > debug_output.mp4) \
>(jq -r '.mediapipe' > mediapipe.json) \
>(jq -r '.fullbody' > fullbody.json) \
>(jq -r '.hand_landmarks' > hand_landmarks.json) \
>(jq -r '.blendshapes' > blendshapes.json) \
>/dev/null
