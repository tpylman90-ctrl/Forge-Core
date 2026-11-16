"""Deterministic Blender procedural object vocabulary for Forge Core."""
import math
import bpy

CATEGORIES = ("crate", "stool", "chair", "mug", "lamp", "hovercraft")


def clear_scene():
    # Keep the camera created by setup_render; remove only generated mesh objects.
    for obj in list(bpy.data.objects):
        if obj.type != "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def material(name, color):
    m = bpy.data.materials.new(name=name)
    m.diffuse_color = color
    return m


def cube(name, loc, dims, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.object
    o.name = name
    o.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if bevel > 0.001:
        b = o.modifiers.new("Edge chamfer", "BEVEL")
        b.width = min(bevel, min(dims)*0.18)
        b.segments = 1
        o.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return o


def cylinder(name, loc, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth, location=loc)
    o=bpy.context.object; o.name=name; o.data.materials.append(mat)
    b=o.modifiers.new("Soft rim","BEVEL"); b.width=min(radius*0.08,depth*0.12); b.segments=1
    o.modifiers.new("Weighted normals","WEIGHTED_NORMAL")
    return o


def cone(name, loc, r1, r2, depth, mat):
    bpy.ops.mesh.primitive_cone_add(vertices=24,radius1=r1,radius2=r2,depth=depth,location=loc)
    o=bpy.context.object; o.name=name; o.data.materials.append(mat)
    o.modifiers.new("Weighted normals","WEIGHTED_NORMAL")
    return o


def torus(name, loc, major, minor, mat, rotation=(0,0,0)):
    bpy.ops.mesh.primitive_torus_add(major_radius=major,minor_radius=minor,major_segments=24,
        minor_segments=8,location=loc,rotation=rotation)
    o=bpy.context.object; o.name=name; o.data.materials.append(mat); return o


def dimensions(params):
    p=list(params)+[0.5]*max(0,8-len(params))
    return tuple(0.45+1.25*max(0.0,min(1.0,float(p[i]))) for i in range(3))


def build_object(category, params, color=(0.42,0.48,0.55,1.0)):
    if category not in CATEGORIES: raise ValueError(f"Unknown category {category}")
    p=list(params)+[0.5]*max(0,8-len(params)); w,d,h=dimensions(p)
    bevel=0.015+0.07*p[3]
    main=material("Main finish",color)
    dark=material("Mechanical trim",(0.07,0.09,0.12,1))
    accent=material("Accent",(0.20+0.35*p[6],0.32,0.58,1))
    if category=="crate":
        cube("Crate body",(0,0,h/2),(w,d,h),main,bevel)
        rib=0.035+p[4]*0.055
        for x in (-w*0.38,w*0.38):
            cube("Vertical rib",(x,-d/2-rib/2,h/2),(rib,rib,h*0.82),dark,rib*0.2)
        cube("Top rim",(0,0,h*0.94),(w*0.94,d*0.94,h*0.07),accent,bevel*0.35)
    elif category in ("stool","chair"):
        seat_h=h*(0.07+p[4]*0.07); seat_z=h*0.48 if category=="chair" else h-seat_h/2
        leg_w=min(w,d)*(0.07+p[5]*0.07)
        cube("Seat",(0,0,seat_z),(w,d*0.92,seat_h),main,bevel)
        for x in (-w*0.39,w*0.39):
            for y in (-d*0.38,d*0.38):
                cube("Leg",(x,y,seat_z/2),(leg_w,leg_w,seat_z),dark,bevel*0.2)
        if category=="chair":
            back_h=h*(0.28+p[6]*0.34)
            cube("Back panel",(0,d*0.40,seat_z+seat_h/2+back_h/2),
                 (w*0.88,leg_w*0.75,back_h),accent,bevel)
    elif category=="mug":
        r=min(w,d)*0.43
        cylinder("Mug body",(0,0,h/2),r,h,main)
        torus("Handle",(r*1.02,0,h*0.53),r*0.52,r*(0.11+p[4]*0.08),accent,(math.pi/2,0,0))
        cylinder("Base ring",(0,0,h*0.045),r*0.88,h*0.07,dark)
    elif category=="lamp":
        base_h=h*(0.06+p[4]*0.04); stem_r=min(w,d)*(0.035+p[5]*0.025)
        cylinder("Lamp base",(0,0,base_h/2),min(w,d)*0.43,base_h,dark)
        cylinder("Lamp stem",(0,0,h*0.48),stem_r,h*0.78,main)
        shade_h=h*(0.20+p[6]*0.12)
        cone("Lamp shade",(0,0,h-shade_h/2),min(w,d)*0.44,min(w,d)*0.26,shade_h,accent)
    elif category=="hovercraft":
        hull_h=h*(0.22+p[4]*0.12)
        cube("Central hull",(0,0,h*0.45),(w*0.62,d,hull_h),main,bevel*1.4)
        pod_w=w*(0.16+p[5]*0.06)
        for x in (-w*0.39,w*0.39):
            cube("Lift pod",(x,-d*0.02,h*0.39),(pod_w,d*0.78,hull_h*0.75),dark,bevel)
            cylinder("Thruster",(x,d*0.48,h*0.43),pod_w*0.30,d*0.16,accent)
        cube("Cockpit",(0,-d*0.12,h*0.69),(w*0.32,d*0.36,h*0.25),accent,bevel*1.5)
        cube("Nose armor",(0,-d*0.40,h*0.48),(w*0.43,d*0.22,hull_h*0.8),main,bevel)
    return (w,d,h)


def setup_render(resolution=96, silhouette=False):
    s=bpy.context.scene
    s.render.engine="BLENDER_WORKBENCH"
    s.render.resolution_x=resolution; s.render.resolution_y=resolution; s.render.resolution_percentage=100
    s.render.image_settings.file_format="PNG"; s.render.image_settings.color_mode="RGB"
    s.render.film_transparent=False
    s.display.shading.light="STUDIO"; s.display.shading.color_type="MATERIAL"
    s.display.shading.show_shadows=True; s.display.shading.show_cavity=True
    s.display.shading.background_type="WORLD"
    s.world.color=(0,0,0) if silhouette else (0.075,0.085,0.10)
    s.view_settings.view_transform="Standard"
    data=bpy.data.cameras.new("ObjectCamera"); cam=bpy.data.objects.new("ObjectCamera",data)
    s.collection.objects.link(cam); s.camera=cam; data.type="ORTHO"
    return cam


def aim_camera(cam,target,direction,scale):
    from mathutils import Vector
    d=Vector(direction).normalized(); t=Vector(target)
    cam.location=t+d*(scale*1.8); cam.rotation_euler=(t-cam.location).to_track_quat("-Z","Y").to_euler()
    cam.data.ortho_scale=scale


def view_directions():
    return {"front":(0,-1,0.25),"side":(1,0,0.22),"top":(0,-0.12,1),"three_quarter":(1,-1,0.42)}
