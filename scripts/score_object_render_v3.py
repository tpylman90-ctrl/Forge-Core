"""Score held-out proxy silhouettes overall and by seen/unseen family."""
import argparse,json
from pathlib import Path
from PIL import Image

HELDOUT={"chair","mug"}
def read_mask(path):
    im=Image.open(path).convert("L")
    pixels=list(im.getdata())
    mask=[v>=128 for v in pixels]
    foreground=sum(mask)/max(1,len(mask))
    if foreground<0.002 or foreground>0.95: raise ValueError(f"Invalid silhouette coverage {foreground:.5f} in {path}")
    return mask,foreground
def main():
    p=argparse.ArgumentParser(); p.add_argument("--pairs",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); pairs=json.loads(a.pairs.read_text())
    scores=[]; coverages=[]; groups={"seen":[],"heldout":[]}
    for pair in pairs:
        target,cov=read_mask(a.pairs.parent/pair["target"])
        pred,_=read_mask(a.pairs.parent/pair["prediction"])
        inter=sum(x and y for x,y in zip(target,pred)); union=sum(x or y for x,y in zip(target,pred))
        if union==0: raise ValueError("Both rendered masks are empty")
        score=inter/union; scores.append(score); coverages.append(cov)
        category=pair["id"].split("_",1)[0]
        groups["heldout" if category in HELDOUT else "seen"].append(score)
    result={"schema":2,"samples":len(scores),"views_per_sample":2,
        "silhouette_iou_mean":sum(scores)/max(1,len(scores)),
        "silhouette_iou_by_family":{k:{"samples":len(v),"mean":sum(v)/max(1,len(v))} for k,v in groups.items()},
        "mean_target_foreground_fraction":sum(coverages)/max(1,len(coverages)),
        "note":"Proxy geometry silhouette overlap; held-out categories are chair and mug. Not photorealism or production topology."}
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
