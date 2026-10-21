# Data preparation and audit

All images are real public benchmark images. No synthetic data substitute was used.
Downloads use immutable repository commits and Git blob verification. Original image files are preserved; training/evaluation decode RGB without resizing.

| Dataset | Count | Role | Source |
|---|---:|---|---|
| CBSD432 | 432 | development split and full formal training | https://github.com/drorsimon/CSCNet/tree/699d08501649979ef487c7e4b5b4a37d0a8ffdf7/datasets/CBSD432/ |
| CBSD68 | 68 | final test only | https://github.com/cszn/FFDNet/tree/e787f46df2f374ff3591186706229cb9a0591fea/testsets/CBSD68/ |
| Kodak24 | 24 | final test only | https://github.com/cszn/FFDNet/tree/e787f46df2f374ff3591186706229cb9a0591fea/testsets/Kodak24/ |
| McMaster | 18 | final test only | https://github.com/cszn/FFDNet/tree/e787f46df2f374ff3591186706229cb9a0591fea/testsets/McMaster/ |

Image manifest SHA256: `85b882a4d2470df9b88ebd3382908fec810f2eba1d27719379c69dc3b97a7935`.
Per-file SHA256, decoded RGB SHA256, dimensions, original mode, source URL, and upstream Git blob SHA1 are recorded in `data/manifest.json`.
Exact file duplicates: 0. Exact decoded RGB duplicates: 0. CBSD432/CBSD68 original image-ID overlap: 0.
The audit does not claim to exclude near-duplicate scenes or differently resized/reencoded versions.

Development split uses the 432 original images before patch extraction: Python `random.Random(2026)` shuffles sorted project-relative paths, assigns 400 to development training and 32 to validation, and writes sorted lists.
Split hashes are in the manifest. Full training uses all 432 original images; the previous 32-image development validation set must not select a formal checkpoint.
Final test datasets are excluded from model selection.

OpenAI RN50 source: https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt. Expected SHA256: `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`. Download verification and actual availability are recorded in `pretrained/RN50.manifest.json`.

Urban100 is prepared separately with `python tools/prepare_data.py --urban100-only`. Its availability, per-image hashes, and combined duplicate audit are recorded in `data/Urban100_manifest.json` when present.
The Urban100 source is the original authors' SelfExSR repository at commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`, `data/Urban100/image_SRF_2/*_HR.png` (100 images). The upstream README explicitly states SR-factor cropping. We use these SRF2 HR files at their supplied resolution without further crop or resize; equivalence to CLIPDenoising authors' unspecified Urban100 copy is unverified. This dataset variant must be disclosed in results.
https://github.com/jbhuang0604/SelfExSR/blob/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/README.md
Alternative author-hosted download attempts: Restormer Google Drive test bundle returned 'Quota exceeded'; IDF OneDrive Urban100 returned HTTP 401. No access restrictions were bypassed.
