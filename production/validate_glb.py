"""Independent stdlib GLB structural/material/budget checks, not visual certification."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from request import load_request

def inspect(path, spec):
    data = Path(path).read_bytes()
    if len(data)<20 or struct.unpack_from("<4sII",data) != (b"glTF",2,len(data)):
        raise ValueError("Invalid GLB header")
    offset=12
    chunks=[]
    while offset<len(data):
        if offset+8>len(data):
            raise ValueError("Truncated GLB chunk")
        length, kind=struct.unpack_from("<I4s",data,offset)
        offset+=8
        if length%4 or offset+length>len(data):
            raise ValueError("Invalid chunk size")
        chunks.append((kind,data[offset:offset+length]))
        offset+=length
    if len(chunks)!=2 or [x[0] for x in chunks]!=[b"JSON",b"BIN\x00"]:
        raise ValueError("Expected JSON and BIN chunks")
    doc=json.loads(chunks[0][1])
    binary=chunks[1][1]
    buffers=doc.get("buffers",[])
    if len(buffers)!=1 or "uri" in buffers[0] or not 0<=len(binary)-buffers[0]["byteLength"]<=3:
        raise ValueError("Invalid embedded buffer")
    views=doc.get("bufferViews",[])
    for view in views:
        start=view.get("byteOffset",0)
        if view["buffer"]!=0 or start<0 or view["byteLength"]<=0 or start+view["byteLength"]>buffers[0]["byteLength"]:
            raise ValueError("Invalid buffer view")
    access=doc.get("accessors",[])
    materials=doc.get("materials",[])
    textures=doc.get("textures",[])
    images=doc.get("images",[])
    if not images or not materials or not textures:
        raise ValueError("Missing textured materials")
    for im in images:
        if "uri" in im or im.get("mimeType") not in ("image/png","image/jpeg"):
            raise ValueError("Texture must be embedded PNG/JPEG")
        v=views[im["bufferView"]]
        start=v.get("byteOffset",0)
        raw=binary[start:start+v["byteLength"]]
        if im["mimeType"]=="image/png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Bad PNG signature")
        if im["mimeType"]=="image/jpeg" and not raw.startswith(b"\xff\xd8"):
            raise ValueError("Bad JPEG signature")
    for t in textures:
        if not 0<=t["source"]<len(images):
            raise ValueError("Invalid texture source")
    triangles=0
    primitive_count=0
    for mesh in doc.get("meshes",[]):
        for p in mesh["primitives"]:
            if p.get("mode",4)!=4 or "indices" not in p:
                raise ValueError("Expected indexed triangles")
            a=p["attributes"]
            if not {"POSITION","NORMAL","TEXCOORD_0"}<=set(a):
                raise ValueError("Missing position, normal or UV")
            pos=access[a["POSITION"]]
            for key in ("min","max"):
                if len(pos.get(key,[]))!=3 or not all(math.isfinite(x) for x in pos[key]):
                    raise ValueError("Missing finite position bounds")
            indices=access[p["indices"]]
            count=indices["count"]
            if count<=0 or count%3:
                raise ValueError("Invalid triangle count")
            m=materials[p["material"]]["pbrMetallicRoughness"]
            if not 0<=m["baseColorTexture"]["index"]<len(textures):
                raise ValueError("Missing basecolor texture")
            triangles+=count//3
            primitive_count+=1
    if not 0<triangles<=spec["triangle_budget"]:
        raise ValueError("Triangle budget failed")
    return {"status":"pass","scope":"structural export checks only","sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"triangles":triangles,"primitive_count":primitive_count,"materials":len(materials),"embedded_images":len(images),"limitations":["No visual-quality certification","No topology/intersection check","No world-space dimension check","No full Khronos conformance validation"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    a=p.parse_args()
    spec=load_request(a.directory/"request.json")
    for name in ("asset.blend","front.png","side.png","top.png","three_quarter.png"):
        if not (a.directory/name).is_file() or (a.directory/name).stat().st_size==0:
            raise ValueError("Missing artifact: "+name)
    report=inspect(a.directory/"asset.glb",spec)
    (a.directory/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main()
