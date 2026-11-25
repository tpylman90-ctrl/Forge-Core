"""Headless deterministic cargo crate prototype. Run only trusted repository code."""
import argparse
import json
import math
import random
import sys
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from request import load_request

def material(name, color, metallic, roughness, size, seed, pattern=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    rng = random.Random(seed)
    # UV-space texture generated as pixels, not a Cycles bake.
    image = bpy.data.images.new(name + "_basecolor", width=size, height=size, alpha=False)
    pixels = []
    for y in range(size):
        for x in range(size):
            grain = rng.uniform(-0.028, 0.028)
            edge = min(x, y, size - 1 - x, size - 1 - y)
            trim = 0.12 if pattern and edge < max(2, size // 64) else 0
            stripe = 0.045 if pattern and (x + y) % max(4, size // 12) == 0 else 0
            pixels.extend([max(0, min(1, c + grain + trim - stripe)) for c in color] + [1.0])
    image.pixels.foreach_set(pixels)
    image.update()
    image.pack()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat

def block(name, pos, dims, mat, radius):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Machined chamfer", "BEVEL")
    bevel.width = min(radius, min(dims) * 0.22)
    bevel.segments = 2
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    # Blender's cube has UVs; smart projection re-packs the chamfer faces too.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.025)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj

def build(spec):
    w, d, h = spec["dimensions_m"]
    u = min(w, d, h)
    n = spec["texture_size"]
    s = spec["seed"]
    body = material("Painted alloy", spec["color"], 0.55, 0.42, n, s, True)
    trim = material("Graphite reinforcement", (0.06,0.075,0.085), 0.35, 0.55, n, s+1)
    accent = material("Safety amber", (0.86,0.36,0.065), 0.2, 0.48, n, s+2, True)
    steel = material("Latch steel", (0.34,0.37,0.39), 0.85, 0.32, n, s+3)
    parts = []
    def add(name, pos, dims, mat, bevel=0.025):
        parts.append(block(name, pos, dims, mat, bevel*u))
    add("Main shell", (0,0,h*0.50), (w*0.90,d*0.90,h*0.80), body, 0.045)
    for side in (-1,1):
        z = h*0.055 if side < 0 else h*0.945
        add("Base frame" if side < 0 else "Lid frame", (0,0,z),(w,d,h*0.11),trim)
        for x in (-w*0.38,w*0.38):
            for y in (-d*0.38,d*0.38):
                add("Corner bumper", (x,y,z),(w*0.19,d*0.19,h*0.11),accent)
    for ysign in (-1,1):
        for x in (-w*0.37,w*0.37):
            add("Face reinforcement",(x,ysign*d*0.465,h*0.5),(w*0.10,d*0.07,h*0.76),trim)
        add("Recess-look inset panel",(0,ysign*d*0.455,h*0.49),(w*0.57,d*0.055,h*0.46),body)
        for x in (-w*0.22,w*0.22):
            add("Latch mount",(x,ysign*d*0.474,h*0.78),(w*0.11,d*0.05,h*0.18),trim)
            add("Latch plate",(x,ysign*d*0.49,h*0.79),(w*0.065,d*0.02,h*0.105),steel)
        for i in range(3):
            add("Warning bar",((i-1)*w*0.065,ysign*d*0.485,h*0.33),(w*0.04,d*0.025,h*0.035),accent,0.008)
    for xsign in (-1,1):
        add("Side inset",(xsign*w*0.458,0,h*0.5),(w*0.065,d*0.65,h*0.48),trim)
        for i in range(4):
            add("Side rib",(xsign*w*0.475,(i-1.5)*d*0.13,h*0.5),(w*0.05,d*0.07,h*0.4),body)
    for y in (-d*0.27,d*0.27):
        add("Lid rail",(0,y,h*0.975),(w*0.76,d*0.09,h*0.05),steel,0.01)
    return parts

def aim(obj, target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()

def render(out, dims):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get("Background").inputs[0].default_value = (0.13,0.15,0.19,1)
    scene.world.node_tree.nodes.get("Background").inputs[1].default_value = 0.45
    w,d,h = dims
    scale = max(dims)
    target = (0,0,h/2)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale*1.9
    scene.camera = camera
    for pos, power, size in [((3,-4,5),900,4),((-3,-1,3),650,3),((1,4,4),1100,3)]:
        bpy.ops.object.light_add(type="AREA", location=tuple(c*scale for c in pos))
        light = bpy.context.object
        light.data.energy = power*scale**2
        light.data.shape = "DISK"
        light.data.size = size*scale
        aim(light,target)
    for name, direction in {"front":(0,-1,0),"side":(1,0,0),"top":(0,0,1),"three_quarter":(1,-1,0.75)}.items():
        camera.location = Vector(target)+Vector(direction).normalized()*scale*5
        aim(camera,target)
        scene.render.filepath = str(out/(name+".png"))
        bpy.ops.render.render(write_still=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request",required=True)
    p.add_argument("--out",required=True)
    args = p.parse_args(sys.argv[sys.argv.index("--")+1:])
    spec = load_request(args.request)
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("Studio")
    bpy.context.scene.unit_settings.system = "METRIC"
    parts = build(spec)
    triangles = 0
    for obj in parts:
        obj.data.calc_loop_triangles()
        triangles += len(obj.data.loop_triangles)
    if triangles > spec["triangle_budget"]:
        raise RuntimeError("Triangle budget exceeded")
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"asset.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out/"asset.glb"),export_format="GLB",use_selection=True)
    (out/"request.json").write_text(json.dumps(spec,indent=2)+"\n")
    render(out,spec["dimensions_m"])
    print("FORGE_EXPORT_COMPLETE",triangles)

if __name__ == "__main__":
    main()
