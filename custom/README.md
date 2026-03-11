# Contextual Image Editor

This repository provides a Python module and command‑line tool for inserting
objects into images in a way that is consistent with the existing scene.  It
demonstrates how to compose **context detection**, **open vocabulary object
detection**, **segmentation** and **inpainting** models into a single
pipeline using open source models hosted on Hugging Face.

## Features

* **Context detection**: The optional `describe_scene` function uses
  **Qwen2.5‑VL** to produce natural language descriptions of an image.  The
  recommended usage of the model with the Hugging Face `transformers` library
  follows the example provided in the model card【107783253617775†L205-L274】.

* **Object detection**: The script uses **Grounding DINO** for zero‑shot
  detection of arbitrary text labels.  `GroundingDinoProcessor` prepares
  image–text pairs and `AutoModelForZeroShotObjectDetection` outputs raw
  predictions, which are post‑processed using the processor’s
  `post_process_grounded_object_detection` method【95351515764910†L194-L249】.

* **Segmentation**: When available, **SAM2** (Segment Anything Model 2) is
  employed to refine the mask of the insertion region.  The script shows how
  to invoke the `mask-generation` pipeline and process bounding boxes with
  `Sam2Processor` and `Sam2Model`【375212058973652†L191-L207】.  If SAM2 is
  unavailable the code falls back to drawing a rectangular mask.

* **Inpainting and insertion**: The final insertion is performed using
  **Stable Diffusion XL Inpainting**.  The `AutoPipelineForInpainting` class
  loads the `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` checkpoint and
  performs masked image editing based on a text prompt, as shown in the model
  card’s example code【835678835445546†L71-L98】.

## Installation

Create a virtual environment and install the dependencies listed in
`requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some of the models are large and will be downloaded from the Hugging Face hub
on first use.  To accelerate downloads you may configure a Hugging Face token
via the `HF_HOME` or `huggingface-cli login` commands.

On systems with a CUDA‑capable GPU, PyTorch will automatically use it if
available; otherwise the pipeline runs on the CPU (which is slower).

## Usage

Invoke the module with an input image, the name of the object to insert and
an output path:

```bash
python contextual_image_editor.py path/to/input.jpg "lemon" path/to/output.jpg
```

The script will:

1. Load the image and optionally describe the scene using Qwen2.5‑VL.
2. Detect existing instances of the specified object with Grounding DINO.
3. Compute a placement for the new object next to the detected items or in
   the centre of the image if none are found.
4. Create a mask over the chosen region (optionally refined by SAM2).
5. Inpaint the masked area with Stable Diffusion XL using a prompt such as
   “a lemon”.

### Example

Suppose you have an image called `fruit.jpg` containing oranges and you want
to add a lemon to the display.  Running:

```bash
python contextual_image_editor.py fruit.jpg lemon out.jpg --device cuda
```

will detect the existing oranges, place a box next to them and generate a
new lemon in that spot.  The output will be saved to `out.jpg`.  If you do
not have a GPU use `--device cpu` (inpainting will take longer).

## Customisation

* **Prompt tuning**: You can edit the prompt construction in
  `contextual_image_editor.py` to control the appearance of the inserted
  object (e.g. “a ripe lemon on a wooden crate”).
* **Mask sizing**: The `compute_insertion_box` function implements a simple
  heuristic.  You can change it to use more sophisticated logic or external
  reasoning for placement.
* **Segmentation**: If you install the optional SAM2 dependencies via
  `transformers>=4.37` the script will produce a more accurate mask.  The
  documentation shows how to supply a bounding box to SAM2 and post process
  the resulting masks【375212058973652†L275-L291】.

## Limitations

* The placement heuristic is simplistic and may not always produce the most
  natural result.  Integrating scene understanding (for example using Qwen
  outputs) or additional geometric reasoning could improve this.
* Stable Diffusion XL sometimes struggles with precise compositional
  placement or perspective.  Fine‑tuning a diffusion model on your specific
  domain or using ControlNet with depth or edge conditions may yield better
  results【771440775016705†L1470-L1513】.
* The pipeline may require several gigabytes of RAM and GPU memory.  Running
  on CPU is possible but slower.

## References

* **Grounding DINO**: Hugging Face documentation on using the model for
  zero‑shot object detection【95351515764910†L194-L249】.
* **SAM2**: Usage examples for automatic mask generation and bounding box
  input【375212058973652†L191-L207】【375212058973652†L275-L291】.
* **Qwen2.5‑VL**: Quickstart for the multimodal model with transformers and
  Qwen utilities【107783253617775†L205-L274】.
* **Stable Diffusion XL Inpainting**: Model card showing how to use the
  AutoPipeline for inpainting【835678835445546†L71-L98】.
* **ControlNet** (optional): Example of conditioning inpainting with Canny
  edges for improved spatial control【771440775016705†L1470-L1513】.