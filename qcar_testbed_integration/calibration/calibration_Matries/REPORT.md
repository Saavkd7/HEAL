# QCar camera calibration package for HEAL

Date: 2026-09-12  
Resolution: 640 x 480 pixels  
Projection model: OpenCV fisheye (`K` plus four coefficients `D`)  
World scale used by HEAL maps: x10

## Decision

The newly measured **front-camera intrinsics** for both QCars pass every
registered quality gate and are approved for an isolated HEAL geometry trial.
The supplied front-only pooling maps were rebuilt from those exact matrices and
their binary layout, BEV index range, valid-point count, and grouped pooling
plans were independently checked.

The maps are deliberately named `trial`: their camera-to-vehicle extrinsic is
the Quanser manufacturer design value, not a new hand-eye/Vicon measurement.
They must therefore be A/B tested against physical/Vicon ground truth before
becoming an accuracy baseline. Copying this package does not activate or
overwrite HEAL's current pooling map.

## Latest front calibration

### QCar qcar-52775 (`192.168.1.198`)

- Images examined/detected/used: 641 / 369 / 344
- Selected model: full fisheye k1-k4
- RMS reprojection error: 0.208335 px
- Point-holdout P95: 0.428794 px
- Checkerboard-corner coverage: 97.348% x 90.787%
- Monotonic/fold-free model: through 90 degrees
- All quality gates: PASS

`K`:

```text
[[319.552724320574,   0.000000000000, 321.403737675700],
 [  0.000000000000, 317.824733807012, 243.315005889003],
 [  0.000000000000,   0.000000000000,   1.000000000000]]
```

`D = [-0.0411626571723, 0.00332319564119, -0.00753684510194, 0.00208963864044]`

Front-only HEAL map:

- Shape/dtype: `[1, 1, 48, 48, 64]`, `int32`
- Valid points: 114,199 / 147,456 (77.45%)
- Occupied BEV cells: 7,498 / 65,536
- Maximum valid index: 65,532; out-of-range indices: 0
- Map/group integrity: PASS

### QCar qcar-52776 (`192.168.1.158`)

- Web calibration completion: `done=true`, 86 captures
- Images detected/used: 83 / 83
- Selected model: full fisheye k1-k4
- RMS reprojection error: 0.246249 px
- Point-holdout P95: 0.429695 px
- Checkerboard-corner coverage: 97.100% x 89.722%
- Monotonic/fold-free model: through 90 degrees
- All quality gates: PASS

`K`:

```text
[[309.788639227712,   0.000000000000, 323.447391460788],
 [  0.000000000000, 308.559188115479, 230.081222708396],
 [  0.000000000000,   0.000000000000,   1.000000000000]]
```

`D = [-0.0440580452052, 0.0221847507664, -0.0192341953288, 0.00455639002877]`

Front-only HEAL map:

- Shape/dtype: `[1, 1, 48, 48, 64]`, `int32`
- Valid points: 113,743 / 147,456 (77.14%)
- Occupied BEV cells: 7,866 / 65,536
- Maximum valid index: 65,532; out-of-range indices: 0
- Map/group integrity: PASS

## Intrinsic file contract

Each latest NPZ contains exactly:

- `front_K`: 3 x 3 float matrix
- `front_D`: four OpenCV-fisheye coefficients
- `image_size`: `[640, 480]`

Use these only with `--projection opencv_fisheye`. Do not pass `D` to
`cv2.undistort` or a five-coefficient Brown/pinhole model.

## Extrinsic used by the trial maps

The shared NPZ contains Quanser manufacturer camera-to-vehicle transforms. The
front transform used here is:

```text
R_camera_to_vehicle = [[ 0,  0,  1],
                       [-1,  0,  0],
                       [ 0, -1,  0]]
t_vehicle_m          = [0.1930, 0.0000, 0.0953]
```

The map builder multiplies translation by ten for HEAL's x10 model world; it
does not scale the intrinsic matrix.

## Package layout and precedence

- `qcar52775/latest_front/` and `qcar52776/latest_front/`: current verified
  front intrinsics and matching one-camera HEAL maps. The only calibration
  this project actually uses (front-only pipeline) or any script reads.
- `shared/`: manufacturer/design extrinsic NPZ.
- `tools/`: exact calibration, verification, map-build, and grouping scripts.
- `SHA256SUMS`: integrity manifest for every copied artifact and script.

Raw checkerboard images are not duplicated into the HEAL repository. They
remain under:

- `Concordia/quality_reports/qcar_coop_runs/20260911_qcar52775_front_calib/`
- `Concordia/quality_reports/qcar_coop_runs/20260912_qcar52776_front_calib/capture/`

## Independent reproduction (2026-09-17)

qcar-52775's front intrinsics above were re-derived from scratch, not just
read from this report: `tools/verify_final_intrinsics.py` was re-run against
the original 1554 raw checkerboard photos (`Concordia/quality_reports/qcar_coop_runs/20260911_qcar52775_front_calib/`),
independently of this package's own numbers. Result matched within a
negligible margin — coverage identical to the thousandth, `K`/`D` within
<0.1%, and one more quality gate passed than originally recorded (6/6 vs.
5/5). qcar-52776's intrinsics (web-calibration workflow, `calib_web.py`, not
this batch script) have **not** been independently re-verified the same way.

## Activation rule

For a one-front-camera run, the HEAL encoder must accept
`[1, 3, 384, 512]`, and the runtime must receive exactly one RGB tensor per
frame. For a full front/left/right run, do not reuse the front matrix for the
other lenses: build a three-camera NPZ/map with camera-specific K/D values.
Keep the currently deployed map as the rollback control and record its SHA-256
before any A/B deployment.

