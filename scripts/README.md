# scripts

`train.py`, `play.py`, `list_envs.py` — thin entry points mirroring
`unitree_rl_mjlab`'s scripts of the same name (minus motion-tracking
support, which no soarm_mjlab task uses).

Example smoke run (CPU, no GPU on this machine so `--gpu-ids None` is
required — the default assumes GPU 0 exists):

```bash
python scripts/train.py SoArm100-Reach --env.scene.num-envs=4 \
    --agent.max-iterations=2 --gpu-ids None
```
