"""Separate mixed Roboflow extractions without modifying originals or metadata."""

from jetson_paths import PROJECT_ROOT, RUNTIME_ROOT
from pathlib import Path
import json,shutil,hashlib
import yaml
home=Path.home();root=home/'robot-perception/datasets'
ball_names=['Break','Cue_Ball','Eight','Five','Four','Nine','Object_Ball','One','Seven','Six','Three','Two']
for name,names in [('pockets',['pocket']),('numbered_balls',ball_names)]:
 target=root/name;manifest=[];counts={}
 for split in ('train','valid','test'):
  out=target/split
  (out/'images').mkdir(parents=True,exist_ok=True);(out/'labels').mkdir(parents=True,exist_ok=True)
  count=0
  for label in sorted((home/split/'labels').glob('*.txt')):
   is_pocket=label.name.startswith('IMG_')
   if is_pocket!=(name=='pockets'):continue
   text=label.read_text()
   for line in text.splitlines():
    values=line.split()
    if values and (len(values)!=5 or not 0<=int(values[0])<len(names)):raise ValueError('Invalid annotation class/format')
   matches=list((home/split/'images').glob(label.stem+'.*'))
   if len(matches)!=1:raise ValueError('Missing or ambiguous image')
   source=matches[0];dest=out/'images'/source.name
   if not dest.exists():dest.symlink_to(source)
   shutil.copy2(label,out/'labels'/label.name)
   manifest.append(dict(split=split,image=str(source),label_sha256=hashlib.sha256(text.encode()).hexdigest()))
   count+=1
  counts[split]=count
 target.mkdir(parents=True,exist_ok=True)
 (target/'data.yaml').write_text(yaml.safe_dump(dict(path=str(target),train='train/images',val='valid/images',test='test/images',nc=len(names),names=names),sort_keys=False))
 (target/'manifest.json').write_text(json.dumps(dict(separation='IMG_ pocket filenames versus numbered sct-mon filenames, corroborated by extraction-time clusters',counts=counts,files=manifest),indent=2))
 print(name,counts)
