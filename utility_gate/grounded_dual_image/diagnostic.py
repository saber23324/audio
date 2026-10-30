"""Fixed previously-inspected VGG-SS map diagnostic; GT used for scoring only."""
from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from relevance_gate.media import decode_audio,causal_window
from utility_gate.grounded_image_pilot.heatmaps import box_map
from .common import ROOT,REPO,read_jsonl,write_json,write_jsonl,sha256,journal
from .grounder import FrozenLocalizer,sham
from .stats import bootstrap
from .media import raw_window

def main():
    g=FrozenLocalizer(torch.device('cpu'))
    rows=read_jsonl(REPO/'artifacts/utility_gate/vggss_grounded_image_pilot_20261028/manifest.jsonl')[:32]
    out=[];directory=ROOT/'maps/diagnostic_known_vggss_dev_rawcrop_v1';directory.mkdir(exist_ok=True)
    for row in rows:
        im=np.asarray(Image.open(REPO/row['frame_path']).convert('RGB'));t=float(row['annotation_time_s'])
        try:
            window,valid,reason,trace=raw_window(row['media_path'],t)
            pred=g.predict(im[None],window[None].numpy(),[valid]);h=pred['maps'][0]
            null=g.predict(im[None],np.zeros((1,32000),np.float32),[True])['maps'][0]
            # Box is deliberately accessed only after both independent maps.
            roi=box_map(row['bbox'],*im.shape[:2]);area=float(roi.mean());shifted=sham(h,row['source_id'],0)
            score={'source_id':row['source_id'],'clip_id':row['clip_id'],'frame_time':t,'frame_pts_s':row['frame_pts'],
                'source_role':'previously inspected VGG-SS development; not new confirmation','map_origin':'EZVSL_PRED','GT_input':False,
                'GT_role':'evaluation mask only, never localizer or VLM input','audio_valid':bool(valid),'map_valid':pred['valid'][0],'roi_area':area,
                'prediction_minus_null_map_MAE':float(np.abs(h-null).mean()),'null_diagnostic_quality_gate_bypassed':True}
            for label,map_ in [('predicted',h),('sham',shifted),('null_waveform',null)]:
                mass=float((map_*roi).sum()/map_.sum()) if map_.sum()>0 else None
                score[label+'_SoundMass']=mass
                score[label+'_area_corrected_mass']=(mass-area)/(1-area) if mass is not None and area<1-1e-8 else None
                peak=np.unravel_index(map_.argmax(),map_.shape)
                score[label+'_pointing_hit']=bool(roi[peak]>.5) if map_.sum()>0 else None
            np.savez_compressed(directory/f"{row['clip_id']}.npz",predicted=h,null_waveform=null,sham=shifted)
            out.append(score)
        except ValueError as e:out.append({'source_id':row['source_id'],'clip_id':row['clip_id'],'status':'invalid_audio','error':str(e)})
    write_jsonl(ROOT/'qa_eval/known_vggss_map_diagnostics_rawcrop_v1.jsonl',out)
    fields=['predicted_SoundMass','sham_SoundMass','null_waveform_SoundMass','predicted_area_corrected_mass','sham_area_corrected_mass','predicted_pointing_hit','prediction_minus_null_map_MAE']
    aggregate={field:bootstrap([r for r in out if r.get(field) is not None and r.get('audio_valid') and r.get('map_valid')],field) for field in fields}
    write_json(ROOT/'qa_eval/known_vggss_map_aggregate_rawcrop_v1.json',{'n_fixed_sources':32,'n_valid':sum(bool(r.get('audio_valid') and r.get('map_valid')) for r in out),'map_metrics':aggregate,'scope':'fixed first32 old development sources, checkpoint and preprocessing not tuned here; not independent localization confirmation; no GT in prediction, no QA benefit inferred'})
    journal('frozen localizer diagnostic on fixed first32 already-inspected VGGSS sources',n_sources=32,GT_used_for_scoring_only=True)
    print(aggregate,flush=True)

if __name__=='__main__':main()
