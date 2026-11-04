# third_party

Downloaded robot models and policies used by the T5-20 physical scenes (`scripts/poc/phys_*`).
Everything here except this file is gitignored; it lives in the main checkout at
`C:/Users/Rohan/dev/Perspective/third_party/`. Fetched 2026-11-04, exactly these files and
nothing else.

| Folder | Source | Revision | Licence | Size |
|---|---|---|---|---|
| `mujoco_menagerie/unitree_go2/` | https://github.com/google-deepmind/mujoco_menagerie, folder `unitree_go2` only (25 files: xml, assets/*.obj, pngs, LICENSE, README) | commit `98a70c569c4c1b6da467a20012f766ee1f8dc521` | BSD-3-Clause (`unitree_go2/LICENSE`, model by Unitree Robotics) | 30.9 MB |
| `unitree_rl_gym/` | https://github.com/unitreerobotics/unitree_rl_gym: `deploy/pre_train/g1/motion.pt`, `deploy/deploy_mujoco/deploy_mujoco.py`, `deploy/deploy_mujoco/configs/g1.yaml`, `resources/robots/g1_description/{g1_12dof.xml, scene.xml}` and the 27 meshes `g1_12dof.xml` references | commit `276801e46c5d433564f24658bac64f254b7d2d4b` | BSD-3-Clause (repository licence) | 25.3 MB |
| `cagataydev__strands-isaaclab-go2-rough-policy/` | https://huggingface.co/cagataydev/strands-isaaclab-go2-rough-policy: `exported/policy.onnx`, `exported/policy.onnx.data`, `params/env.yaml`, `params/agent.yaml`, `LICENSE.md`, `README.md` | revision `01679659835cb06576398cc004b15483acbd51d5` | Weights CC-BY-4.0 (attribution: "cagataydev / strands-robots Isaac Lab showcase (strands-labs/robots#4227)"); `params/*.yaml` BSD-3-Clause (Isaac Lab). Allows public demo use with attribution. | 1.2 MB |
| `derived/` | made here by `scripts/poc/export_policies.py` (numpy weight files of the two policies above) | - | as their sources | 1.3 MB |

The `onnxruntime` 1.31.1 pip package was installed into the lerobot-rocm venv only (used once,
to check the numpy port of the Go2 policy); the project env does not need it.

Any public use of the videos that show the Go2 walking should carry the CC-BY attribution
above (e.g. in the description or credits).
