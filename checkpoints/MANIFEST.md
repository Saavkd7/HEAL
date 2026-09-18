# HEAL camera-only checkpoints — manifest

Source: https://huggingface.co/yifanlu/HEAL (fetched 2026-09-09T12:25:10-04:00)
Downloaded read-only with curl. Blobs are gitignored; this file is not.

**`subdir` column reorganized 2026-09-18** — see `README.md` for the
`qcar/` vs `heal_reference/` split by actual QCar relevance. Original
`.zip`s for already-extracted runs were deleted (pure duplicates); this
table's checksums still describe them if a re-download is ever needed.

| file | subdir | bytes | sha256 |
|---|---|---|---|
| HeterBaseline_opv2v_camera_attfuse_2023_08_08_16_50_01.zip | qcar | 289318324 | 1a071c5c073be10abcc23b1ccb4301f7ec34baf757f246561b9d0db8f6dc3247 |
| HeterBaseline_opv2v_camera_disco_2023_08_08_16_50_01.zip | heal_reference/opv2v_camera | 1431693665 | 57253a24263a0ca3cc3377c221a959984c9b8ccfcf029f9775134191f3a63b23 |
| HeterBaseline_opv2v_camera_fcooper_2023_08_06_11_48_21.zip | heal_reference/opv2v_camera | 295644584 | 944a9eccdb974d56dff99b9a74ed7cc5ac5b8f83bf5ebf1a2d07b483e5bf0afa |
| HeterBaseline_opv2v_camera_v2xvit_2023_08_07_04_46_10.zip | heal_reference/opv2v_camera | 347333965 | 58022b0c93f2346ffc423c7ed15bcaa9b656db9b8059f60a77d56ab116dd12ae |
| HeterBaseline_DAIR_camera_attfuse_2023_09_09_11_24_40.zip | heal_reference/dair_camera | 342501198 | 031af91e71b2fc59bfc58e07842829084b51c08847a76925bb97ad697021084b |
| HeterBaseline_DAIR_camera_cobevt_2023_09_09_11_25_45.zip | heal_reference/dair_camera | 325819151 | a67740a76875eb7b25bf42c6eaf18382493ec9fa57d42b436aa1d3308612208a |
| HeterBaseline_DAIR_camera_disco_2023_09_09_11_27_56.zip | heal_reference/dair_camera | 305841947 | ec3aa68a08bdab7ac6f958487d2ca337472437eaa932220000e03aab42a92889 |
| HeterBaseline_DAIR_camera_fcooper_2023_09_09_11_28_21.zip | heal_reference/dair_camera | 318149550 | f7e96d20d613888b31b55c5cc2d94ccf9c38121d6ebaf406b007273c229f1121 |
| HeterBaseline_DAIR_camera_v2xvit_2023_09_09_11_27_38.zip | heal_reference/dair_camera | 346770771 | 692dc1b14d22499d6ce9b97a1526085a48debf70bc07751e35b0fbab2456ebc4 |
