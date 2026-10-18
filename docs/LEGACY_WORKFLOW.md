# Original release reference

The maintained workflow is documented in the [root README](../README.md) and
uses `drone-auth`. Historical entry points and their environment are preserved
separately on the public [`legacy/acsac2022` branch](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/legacy/acsac2022),
frozen at commit `88d0098b189b8cb48da3ec05ab81f7890b36d6b5`.

Use the fixed snapshot links below to inspect the original implementation:

- [Original README, experiment commands and tutorial images](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/blob/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/README.md)
- [Original Conda environment](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/blob/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/environment.yml)
- [PKL dataset generation](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/dataset_build)
- [Experiment scripts](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/experiment)
- [Training and evaluation runners](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/runners)
- [Historical YAML configurations](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/config)
- [Original toolbox](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5/toolbox)

The maintained tree no longer includes these historical scripts, their YAML
configurations, `environment.yml`, attack tutorial screenshots, or the
`pkl_dataset/`, `trained_model/` and `results/` output placeholders. The shared
MFCC core and its license remain in `toolbox/` because the maintained workflow
uses them. The old `toolbox.CMatrix` export is also removed; current plotting
uses `drone-auth plot` and `src/drone_fingerprint/plotting.py`.

For historical investigation, use a separate checkout of the frozen branch and
follow its own README. Modern environment tests do not validate the historical
environment or guarantee that its original dependencies remain installable.
Keep generated artifacts outside the maintained checkout. See
[release status](RELEASE_STATUS.md) for version boundaries and
[numerical validation](NUMERICAL_VALIDATION.md) for the maintained definitions.
