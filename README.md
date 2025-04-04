# Mediapipe Skeleton to COCO JSON
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

**WORK IN PROGRESS**

This is a Mediapipe workflow that writes the Mediapipe skeleton, including hands, into a COCO-style JSON format. The workflow utilizes the Mediapipe framework, which provides a flexible and efficient infrastructure for building multimodal applied machine learning models.

[Now on replicate -- live.](https://replicate.com/fire/v-sekai.mediapipe-labeler)

<img src="Screenshot 2023-09-03 065500.png" width="25%"> <img src="thirdparty/image.jpg" width="25%">

**Model File**: [Link to Model Card Blendshape V2.pdf](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)

**COCO Keypoints File**: https://roboflow.com/formats/coco-keypoint

# Install

```zsh
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
# Generate COCO-style JSON with Mediapipe skeleton and hands.
cog predict -i image_path=@./thirdparty/image.jpg
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://chibifire.com"><img src="https://avatars.githubusercontent.com/u/32321?v=4?s=100" width="100px;" alt="K. S. Ernest (iFire) Lee"/><br /><sub><b>K. S. Ernest (iFire) Lee</b></sub></a><br /><a href="#research-fire" title="Research">🔬</a> <a href="https://github.com/V-Sekai/V-Sekai.mediapipe-labeler/commits?author=fire" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
