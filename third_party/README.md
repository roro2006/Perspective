# third_party

Downloaded robot models and policies used by the T5-20 physical scenes (`scripts/poc/phys_*`).
Everything here except this file is gitignored; it lives in the main checkout at
`C:/Users/Rohan/dev/Perspective/third_party/`. The code looks in `$CURIO_THIRD_PARTY`, then
this checkout's `third_party/`, then that path (scripts/poc/phys_core.py). Fetched 2026-11-04 (T5-32's row 2026-11-05),
exactly these files and nothing else.

| Folder | Source | Revision | Licence | Size |
|---|---|---|---|---|
| `mujoco_menagerie/unitree_go2/` | https://github.com/google-deepmind/mujoco_menagerie, folder `unitree_go2` only (25 files: xml, assets/*.obj, pngs, LICENSE, README) | commit `98a70c569c4c1b6da467a20012f766ee1f8dc521` | BSD-3-Clause (`unitree_go2/LICENSE`, model by Unitree Robotics) | 30.9 MB |
| `unitree_rl_gym/` | https://github.com/unitreerobotics/unitree_rl_gym: `deploy/pre_train/g1/motion.pt`, `deploy/deploy_mujoco/deploy_mujoco.py`, `deploy/deploy_mujoco/configs/g1.yaml`, `resources/robots/g1_description/{g1_12dof.xml, scene.xml}` and the 27 meshes `g1_12dof.xml` references | commit `276801e46c5d433564f24658bac64f254b7d2d4b` | BSD-3-Clause (repository licence) | 25.3 MB |
| `cagataydev__strands-isaaclab-go2-rough-policy/` | https://huggingface.co/cagataydev/strands-isaaclab-go2-rough-policy: `exported/policy.onnx`, `exported/policy.onnx.data`, `params/env.yaml`, `params/agent.yaml`, `LICENSE.md`, `README.md` | revision `01679659835cb06576398cc004b15483acbd51d5` | Weights CC-BY-4.0 (attribution: "cagataydev / strands-robots Isaac Lab showcase (strands-labs/robots#4227)"); `params/*.yaml` BSD-3-Clause (Isaac Lab). Allows public demo use with attribution. | 1.2 MB |
| `kouylty__Isaac-Go2-Navigation-Rough-v0/` | https://huggingface.co/kouylty/Isaac-Go2-Navigation-Rough-v0 (T5-32): `README.md`, `locomotion/policy.pt`, `locomotion/model_1998.pt`, `navigation/policy.pt`, `navigation/model_2499.pt` | revision `6c61ab2b7e7428c0edfe3f0febc9c6d49a989018` | MIT (repository card; by kouylty, trained with NVIDIA Isaac Lab (BSD-3-Clause) on the Unitree Go2) | 10.7 MB |
| `derived/` | made here by `scripts/poc/export_policies.py` (numpy weight files of the two policies above) and `scripts/poc/export_go2_nav.py` (T5-32: `go2_nav_loco.npz`, `go2_nav_nav.npz`, `go2_nav_samples.npz`) | - | as their sources | 2.8 MB |

The `onnxruntime` 1.31.1 pip package was installed into the lerobot-rocm venv only (used once,
to check the numpy port of the Go2 policy); the project env does not need it.

Any public use of the videos that show the Go2 walking should carry the CC-BY attribution
above (e.g. in the description or credits).

The T5-32 navigation and walking policies are kouylty's (MIT); credit "kouylty /
Isaac-Go2-Navigation-Rough-v0, trained in NVIDIA Isaac Lab" and Unitree Robotics for the Go2
when showing them.

**Demo laptop (T5-34, 2026-11-06).** For the live dog demo, only the Go2's two rows were fetched
again into `C:/Users/Harry/Desktop/MoreProjects/Perspective/third_party/`, at the same pinned
revisions: Menagerie `unitree_go2/` (all 25 files, through the GitHub tree API at
`98a70c5`, 30 MB) and the six cagataydev files at `0167965` (1.2 MB).
`export_policies.py --go2` wrote `derived/go2_rough.npz`; its check against onnxruntime
(installed only for that one run, with `uv run --with onnxruntime`, not in the project env)
gave max |numpy - onnxruntime| 3.3e-6, as on the GPU box. No G1 or T5-32 files are on the laptop.
