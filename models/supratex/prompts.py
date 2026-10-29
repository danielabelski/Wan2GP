# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

TEXTURE_INFOS = """### SUPRA-Tex Texturer
Paints a mesh with **PBR textures** (base colour, metallic, roughness) from **one picture of the object**. The colours seen in the picture are kept on the surfaces the camera sees, with their fine print and patterns; the rest of the object is completed consistently by a 3D model, so the back matches the front's materials.

### Inputs
- **Input Mesh**: a .glb, .obj, .ply or .stl mesh. Its geometry is kept (only reduced when **Max Faces** is set).
- **Object Image**: the same object, seen from the front of the mesh. A transparent background is best; other images have their background removed.

The mesh is placed automatically to match the picture's silhouette. It works best when the picture shows the object as the mesh was made from it, for instance the image given to Hunyuan3D, TRELLIS.2 or Hi3D. A mesh that differs from the picture gets the picture's colours only where both agree; other regions are completed.

### Settings
- **Steps** (default 25): completion steps of the texture.
- **Max Faces**: 0 keeps every face (a mesh refined by Hi3D keeps its detail, unwrapping takes longer); otherwise the mesh is reduced first.
- **Texture Size**: 2048 by default, 4096 for close-ups.

### Combining
Use it as the **Texturer** of the post-processing to paint the meshes of untextured 3D models, after the Hi3D 3D Refiner if you like: shape, then fine geometry, then textures.

### License
The material stage uses the SUPRA-Tex Material Adapter for Qwen Image 2.1, released under the **Qwen Research License: non-commercial use only**."""

TEXTURE_PROMPT_INFOS = """The prompt only names the output files (for example "Leather Armchair"): the textures come from the Object Image."""

TEXTURE_DEEPY_INFOS = ("Textures an existing mesh (custom_guide: .glb/.obj/.ply/.stl) with PBR base colour, metallic and roughness from one object image "
                       "(image_refs, video_prompt_type 'I'). The visible side keeps the image's colours and print; hidden sides are completed in 3D. The mesh is "
                       "placed automatically from its silhouette, so the image must show the same object facing the mesh front (+Z): ideal for the image a mesh was "
                       "generated from. num_inference_steps 25 (completion steps); custom_settings.max_faces 0 keeps every face; custom_settings.texture_size "
                       "1024/2048/4096. Also available as the 'supratex' Texturer of 3D post-processing (temporal_upsampling slot), alone or after the 'hi3d' 3D Refiner. "
                       "The material adapter is non-commercial (Qwen Research License).")

TEXTURE_DEEPY_PROMPT_INFOS = "The prompt only names the output files; leave it short (an asset name)."

SHAPE_INFOS = """### Pixal3D Shape
Turns **one picture of an object** into a 3D mesh (GLB, geometry only) that is **aligned with the picture**: the mesh is built in the camera of the picture, so its visible side matches the image pixel for pixel. This makes it a good base for texturing from the same picture.

### Inputs
- **Object Image**: one object; a transparent background is best, other images have their background removed. The camera's field of view is estimated from the picture.

### Settings
- **Steps** (default 12) and **Guidance** (default 7.5) of the three geometry stages (structure, coarse shape, fine shape up to 1536).
- **Max Faces**: the mesh is reduced to this face count (0 keeps all).

The hidden side is inferred; thin parts and objects seen at an unusual angle may come out less complete."""

SHAPE_PROMPT_INFOS = """The prompt only names the output files (for example "Pitcher Plant"): the shape comes from the Object Image."""

SHAPE_DEEPY_INFOS = ("Pixal3D: one object image (image_refs, video_prompt_type 'I') to an untextured, pixel-aligned mesh (GLB): the mesh is generated in the "
                     "picture's camera (field of view estimated by MoGe-2), so it fits the picture exactly; a good base for SUPRA-Tex texturing. "
                     "num_inference_steps 12 and guidance_scale 7.5 per geometry stage; custom_settings.max_faces (default 500000, 0 keeps all).")

FULL_INFOS = """### SUPRA-Tex (Pixal3D + SUPRA-Tex)
Creates a **textured 3D object from one picture**, or textures an existing mesh.

### Modes
- **Shape + PBR Texture** (default): Pixal3D builds a mesh aligned with the picture, then SUPRA-Tex paints it with PBR textures (base colour, metallic, roughness). The visible side keeps the picture's colours and print exactly; the hidden side is completed in 3D.
- **Shape Only**: the Pixal3D mesh alone.
- **Texture a Mesh**: SUPRA-Tex textures the **Input Mesh** from the picture; the mesh is placed automatically to match the picture's silhouette, so the picture must show the same object.

### Settings
- **Steps** (default 12) and **Guidance** (default 7.5): the geometry stages; the texture always uses 25 completion steps.
- **Max Faces**: the generated mesh is reduced to this face count (Texture a Mesh: 0 keeps every face of the Input Mesh).
- **Texture Size**: 2048 by default, 4096 for close-ups.

### License
The material stage uses the SUPRA-Tex Material Adapter for Qwen Image 2.1, released under the **Qwen Research License: non-commercial use only**."""

FULL_DEEPY_INFOS = ("Pixal3D geometry + SUPRA-Tex PBR texturing from one object image (image_refs, video_prompt_type 'I'). model_mode 'shape_texture' (default): "
                    "pixel-aligned mesh then PBR textures (base colour, metallic, roughness) keeping the image's colours and print on the visible side, the hidden "
                    "side completed in 3D; 'shape': the mesh only; 'texture': textures custom_guide (an existing .glb/.obj/.ply/.stl mesh, placed from its "
                    "silhouette). num_inference_steps 12 and guidance_scale 7.5 for the geometry; custom_settings.max_faces (default 500000), "
                    "custom_settings.texture_size 1024/2048/4096. The material adapter is non-commercial (Qwen Research License).")
