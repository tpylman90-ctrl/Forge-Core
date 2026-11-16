"""Blender silhouette evaluation for held-out predicted procedural object specs."""
import argparse,json,sys
from pathlib import Path
import bpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from object_geometry_v1 import aim_camera,build_object,clear_scene,setup_render,material

def render_mask(cam,cat,params,dims,direction,res):
    clear_scene(); build_object(cat,params,(1,1,1,1))
    white=material("silhouette",(1,1,1,1))
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            obj.data.materials.clear(); obj.data.materials.append(white)
    scene=bpy.context.scene; scene.world.color=(0,0,0)
    aim_camera(cam,(0,0,dims[2]*0.5),direction,max(dims)*1.9)
    scene.render.resolution_x=res; scene.render.resolution_y=res; scene.render.resolution_percentage=100
    bpy.ops.render.render()
    px=list(bpy.data.images["Render Result"].pixels)
    return [1 if (px[i]+px[i+1]+px[i+2])/3>0.45 else 0 for i in range(0,len(px),4)]

def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int,default=72)
    ap.add_argument("--resolution",type=int,default=96); a=ap.parse_args(raw); a.out.mkdir(parents=True,exist_ok=True)
    cam=setup_render(a.resolution,True); rows=json.loads(a.predictions.read_text())[:a.limit]; scores=[]
    views=(("front",(0,-1,0.25)),("three_quarter",(1,-1,0.42)))
    for row in rows:
        dims=[0.45+1.25*float(row["target_params"][i]) for i in range(3)]; per=[]
        for _,direction in views:
            target=render_mask(cam,row["target_category"],row["target_params"],dims,direction,a.resolution)
            pred=render_mask(cam,row["predicted_category"],row["predicted_params"],dims,direction,a.resolution)
            inter=sum(x and y for x,y in zip(target,pred)); union=sum(x or y for x,y in zip(target,pred))
            per.append(inter/max(1,union))
        scores.append(sum(per)/len(per))
    result={"schema":1,"samples":len(scores),"views_per_sample":2,
        "silhouette_iou_mean":sum(scores)/max(1,len(scores)),
        "silhouette_iou_by_sample":[round(float(x),6) for x in scores],
        "note":"Proxy geometry silhouette overlap; not a photorealistic render or production-topology score."}
    (a.out/"render_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
