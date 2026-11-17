"""Render target and predicted proxy geometry to PNG masks for host-side IoU scoring."""
import argparse,json,sys
from pathlib import Path
import bpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from object_geometry_v1 import aim_camera,build_object,clear_scene,setup_render,material

def render_png(cam,category,params,dims,direction,path,resolution):
    clear_scene()
    build_object(category,params,(1,1,1,1))
    white=material("Silhouette",(1,1,1,1))
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            obj.data.materials.clear()
            obj.data.materials.append(white)
    scene=bpy.context.scene
    scene.world.color=(0,0,0)
    aim_camera(cam,(0,0,dims[2]*0.5),direction,max(dims)*1.9)
    scene.render.resolution_x=resolution
    scene.render.resolution_y=resolution
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGB"
    scene.render.filepath=str(path)
    bpy.ops.render.render(write_still=True)

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=96)
    ap.add_argument("--resolution",type=int,default=96)
    a=ap.parse_args(raw); a.out.mkdir(parents=True,exist_ok=True)
    cam=setup_render(a.resolution,True)
    rows=json.loads(a.predictions.read_text())[:a.limit]
    views=(("front",(0,-1,0.25)),("three_quarter",(1,-1,0.42)))
    pairs=[]
    for i,row in enumerate(rows):
        dims=[0.45+1.25*float(row["target_params"][j]) for j in range(3)]
        for view,direction in views:
            base=f'{i:04d}_{row["id"]}_{view}'
            target=a.out/f"{base}_target.png"
            pred=a.out/f"{base}_pred.png"
            render_png(cam,row["target_category"],row["target_params"],dims,direction,target,a.resolution)
            render_png(cam,row["predicted_category"],row["predicted_params"],dims,direction,pred,a.resolution)
            pairs.append({"id":row["id"],"family_split":row.get("family_split","seen"),"view":view,"target":target.name,"prediction":pred.name})
        if (i+1)%24==0: print(json.dumps({"event":"rendered_test_objects","count":i+1}),flush=True)
    (a.out/"render_pairs.json").write_text(json.dumps(pairs,indent=2)+"\n")
    print(json.dumps({"event":"mask_renders_complete","pairs":len(pairs)},indent=2),flush=True)

if __name__=="__main__": main()
