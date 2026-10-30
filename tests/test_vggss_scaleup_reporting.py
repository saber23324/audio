import tempfile
import json
from pathlib import Path
import unittest

from utility_gate.grounded_image_pilot.report import mean_ci
from utility_gate.grounding.scaleup_report import decisions,historical_partition,median_ci,merge_holdout,reference_summary,weighted_ci
from utility_gate.common import sha256,write_json,write_jsonl,read_jsonl


def metric(mean,lo,hi):return {'mean':mean,'ci95':[lo,hi],'n':100}


def fixture(mass=.001,lo=-.001,hi=.004,gold=.01):
    contrast={'spatial':{'mass':metric(mass,lo,hi)},'category_proxy':{'n':100,'gold_nll':metric(gold,gold-.001,gold+.001)},
              'teacher_fidelity':{'kl_teacher_to_student':metric(.02,.01,.03)}}
    bias={'spatial':{'mass':metric(.12,.11,.13)},'category_proxy':{'n':100,'gold_nll':metric(-.01,-.02,-.005)}}
    return {'gt_overlay_minus_rgb_clean':contrast,'gt_overlay_minus_sham_overlay':contrast,'gt_routing_bias_minus_rgb_clean':bias},\
           {'gt_routing_bias':{'features':{'compressed_tokens_Z.relative_l2_global':{'mean':.05}}}}


class ScaleupReports(unittest.TestCase):
    def test_history_scope_excludes_only_current_run_even_for_relative_cli(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();other=root.parent/(root.name+'_historical')
            snapshot={str(root/'STATUS.json'):'new',str(other/'STATUS.json'):'old'}
            old,current=historical_partition(snapshot,Path(tmp))
            self.assertEqual(list(current),[str(root/'STATUS.json')]);self.assertEqual(list(old),[str(other/'STATUS.json')])

    def test_shards_merge_one_locked_cohort_without_duplicate_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifests').mkdir()
            rows=[{'source_id':str(i),'clip_id':str(i)} for i in range(4)]
            manifest=root/'manifests/fresh_holdout_LOCKED.jsonl';write_jsonl(manifest,rows)
            write_json(root/'pretest_lock.json',{'holdout_manifest_sha256':sha256(manifest),'inference_source_sha256':{}})
            for shard in range(2):
                label='fresh_holdout_shard'+str(shard);p=root/'per_source'/label;p.mkdir(parents=True)
                write_jsonl(p/'results.jsonl',rows[shard::2]);write_jsonl(p/'stage3_references.jsonl',rows[shard::2])
                f=root/'features'/label;f.mkdir(parents=True);(f/(str(shard)+'.npz')).write_bytes(bytes([shard]))
            merge_holdout(root)
            self.assertEqual(read_jsonl(root/'per_source/fresh_holdout/results.jsonl'),rows)
            with self.assertRaises(FileExistsError):merge_holdout(root)

    def test_oracle_bias_lead_is_distinct_from_Overlay_go(self):
        pairs,methods=fixture();value=decisions(pairs,methods,500)
        self.assertFalse(value['GO_A_RGB_Overlay']);self.assertTrue(value['GO_B_predicted_bias_research_lead'])
        self.assertTrue(value['NO_GO_current_Overlay'])

    def test_positive_spatial_and_NLL_evidence_needed_for_go_A(self):
        pairs,methods=fixture(.02,.015,.025,-.01);value=decisions(pairs,methods,500)
        self.assertTrue(value['GO_A_RGB_Overlay']);self.assertFalse(value['NO_GO_current_Overlay'])
        self.assertIn('not_established',value['GT_free_audio_gain'])

    def test_wide_intervals_are_inconclusive_not_equivalence(self):
        pairs,methods=fixture(.001,-.02,.03,.01);value=decisions(pairs,methods,500)
        self.assertTrue(value['INCONCLUSIVE']);self.assertFalse(value['NO_GO_current_Overlay'])

    def test_uniform_weights_reproduce_source_bootstrap_mean(self):
        x=[-.01,.03,.02,-.04,.005]
        a=weighted_ci(x,[1]*len(x));b=mean_ci(x)
        self.assertAlmostEqual(a['mean'],b['mean'])
        for x,y in zip(a['ci95'],b['ci95']):self.assertAlmostEqual(x,y)

    def test_median_empty_does_not_fabricate_zero(self):
        self.assertIsNone(median_ci([])['median']);self.assertEqual(median_ci([2,2,2])['median'],2.)

    def test_all_invalid_audio_references_remain_unrun_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'per_source/fresh_holdout';p.mkdir(parents=True)
            (p/'stage3_references.jsonl').write_text('{"clip_id":"a","audio_valid":false,"audio_quality":"missing_audio","methods":{}}\n')
            value=reference_summary(root,'fresh_holdout',[{'clip_id':'a'}])
            self.assertEqual(value['planned'],1);self.assertIsNone(value['same_stage3_real_minus_null'])


if __name__=='__main__':unittest.main()
