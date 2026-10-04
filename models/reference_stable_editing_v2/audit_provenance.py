"""P0 source-provenance lock for duplicate physical PSG rows."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--psg',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); src=json.loads(a.psg.read_text()); by=collections.defaultdict(list)
    for row in src['data']: by[row['file_name']].append(row)
    groups=[]
    for fn, rows in sorted(by.items()):
        if len(rows)<2: continue
        pair=[]
        for row in rows:
            pair.append({'image_id':str(row.get('image_id')), 'coco_image_id':str(row.get('coco_image_id')),
                         'pan_seg_file_name':row.get('pan_seg_file_name'),
                         'segments_info_ids':[x.get('id') for x in row.get('segments_info',[])],
                         'relation_count':len(row.get('relations',[])),
                         'annotation_count':len(row.get('annotations',[])),
                         'gqa_category_ids':[x.get('gqa_category_id') for x in row.get('segments_info',[])],
                         'attribute_ids':[x.get('attribute_ids',[]) for x in row.get('segments_info',[])]})
        groups.append({'file_name':fn,'rows':pair,'same_coco_id':len({x['coco_image_id'] for x in pair})==1,
                       'same_panoptic_file':len({x['pan_seg_file_name'] for x in pair})==1,
                       'same_segment_ids':len({tuple(x['segments_info_ids']) for x in pair})==1,
                       'row_lineage_fields_present':False})
    report={'schema_version':1,'name':'reference_stable_editing_v2_source_provenance','status':'unresolved_provenance',
            'source':{'psg':str(a.psg),'dataset_sha256':sha256(a.psg)},
            'population':{'duplicate_file_groups':len(groups),'duplicate_rows':sum(len(x['rows']) for x in groups)},
            'evidence':{'official_dataset_fields_checked':['image_id','coco_image_id','file_name','pan_seg_file_name','segments_info','relations','annotations'],
                        'row_lineage_fields_checked':['annotator_id','annotation_id','annotation_version','source_record_id','timestamp'],
                        'row_lineage_fields_present':False,
                        'same_coco_id_groups':sum(x['same_coco_id'] for x in groups),
                        'same_panoptic_file_groups':sum(x['same_panoptic_file'] for x in groups),
                        'same_segment_id_groups':sum(x['same_segment_ids'] for x in groups)},
            'decision':'HOLD_REFERENCE_STORY',
            'reason':'PSG rows expose no annotator/version/lineage field; duplicate rows are paired annotation records, but independent-valid-reference provenance is not established.',
            'groups':groups}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'decision':report['decision'],'groups':len(groups),'lineage_present':False},indent=2))
if __name__=='__main__': main()
