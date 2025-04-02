media_path="thirdparty/image.jpg"
cog predict -i "media_path=@$media_path" | jq -r '.debug_media | split(",")[1]' | base64 -d > debug_output.png
