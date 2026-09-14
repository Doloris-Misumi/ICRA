# Evaluation protocol

The main table uses a 10,065-frame nonempty-Sedan subset of the 17,536-frame K-Radar test split. This is not an evaluation over all official test frames. `resources/split/test_10065.txt` fixes membership and the original evaluation ordering. The adapter retains the dataset loader's ordering and selects exactly this membership.

Annotations: v1 `info_label`, LiDAR-to-radar x/y calibration and z offset 0.7 m; Sedan only; inclusive center ROI x=[0,72], y=[-6.4,6.4], z=[-2,6]; no additional azimuth filter. Box dimensions in label files are doubled. There are 18,874 retained objects. The preparation script was checked for exact equality with the source experiment's annotations, including class, box, track information and availability.

Scoring: supplied legacy K-Radar evaluator, R11 AP; confidence threshold 0.3; NMS overlap threshold 0.1; report BEV and 3D AP at IoU 0.3 and 0.5. The inherited conversion writes center coordinates without a half-height shift; the inherited 3D overlap implementation uses its legacy height-origin convention. This intentionally reproduces the paper table, not the later corrected evaluator. Scores from other coordinate/AP conventions must not be mixed into this table.

The `USE_CLEAN_EVALUATOR` configuration key names a failure-tracking/empty-frame handling path; it does not certify training-data independence or identify the later corrected geometric evaluator.

Expected epoch-17 overall AP (original main-table implementation): AP3D@0.3=77.8111, AP3D@0.5=47.1809, APBEV@0.3=82.7101, APBEV@0.5=72.9651. Epoch-10: 73.5148, 42.8133, 79.3624, 68.5829, respectively. Small numerical/environment differences can affect inference; these are recorded experiment results, not results of a new run of this release.

Training uses 14,149 internal training frames and 3,309 validation frames. The shared Stage 1 checkpoint's exact producing splits are not established. Test-set results have been inspected during development. The release does not certify a blind test or a full-training-set refit.
