# moss_image_compat

ComfyUI custom node that runs the MOSS-Image node pack
(`submodules/MOSS-Image/comfyui/moss-image`) on ComfyUI versions older than the pack, such
as 3216c62.

The pack needs two things that ComfyUI 3216c62 does not have: the `comfy.storage` module and a
`fast_disk` argument on `ModelPatcher`. This node adds only what is missing, then loads the pack,
so the nodes are the pack's own (MOSS-Image Load Diffusion Model, Load Qwen3.5, the HF loaders).
On a ComfyUI that already has both, it patches nothing.

- `comfy.storage` is a stand-in that reports no model on a fast disk. 3216c62 behaves the same
  way for every model it loads. Its global `--fast-disk` flag still works.
- `fast_disk=` is accepted and then dropped by `ModelPatcher` and `ModelPatcherDynamic`.

## Install

Put **one** of these in `ComfyUI/custom_nodes`, never both, because they register the same nodes:

- this directory, on an older ComfyUI;
- the pack itself, on a ComfyUI new enough to run it.

The node looks for the pack in this order:

1. the directory named by `MOSS_IMAGE_NODE_PATH` (the pack's `comfyui/moss-image` folder). If
   the variable is set and wrong, the node fails with an error. It does not fall back to
   another copy.
2. `moss-image/` inside this directory. For a self-contained install, for example on an offline
   server, copy the pack here:
   ```
   cp -r comfyui/moss_image_compat ComfyUI/custom_nodes/
   cp -r submodules/MOSS-Image/comfyui/moss-image ComfyUI/custom_nodes/moss_image_compat/
   ```
3. the diffusion-pipe checkout this directory belongs to. This works through a symlink:
   ```
   ln -s /path/to/diffusion-pipe/comfyui/moss_image_compat ComfyUI/custom_nodes/
   ```

LoRA and LoKr files trained with diffusion-pipe (`model_type = 'moss_image'`) load with ComfyUI's
standard LoraLoader on top of the MOSS-Image model.

Tests: `python -m pytest -q test/test_moss_image_compat.py`. The end-to-end test loads the node
the way ComfyUI does. Run it at both ComfyUI pins.
