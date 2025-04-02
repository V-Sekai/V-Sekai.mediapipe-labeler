# With proper filename quoting and base64 decoding
media_path="@【まなこ×やっこ】CALL ME CALL ME　踊ってみた【オリジナル振付】 [75izBsrw-sw].mp4"
cog predict -i "media_path=$media_path" | jq -r '.debug_media | split(",")[1]' | base64 -d > debug_output.mp4
