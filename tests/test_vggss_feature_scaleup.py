import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from utility_gate.grounding.feature_diagnostics import FeatureCapture
from utility_gate.grounding.feature_stats import compression_decomposition,feature_response,pixel_response,routing_response
from utility_gate.grounding.scaleup_manifest import collect_evidence,holdout_guard,identity,validate_pools
from utility_gate.grounding.scaleup_data import proportional_sample
from utility_gate.grounding.feature_viz import summarize
from collections import defaultdict


class FeatureResponses(unittest.TestCase):
    def test_identical_features_have_exact_zero_delta(self):
        x=torch.randn(1,7,6);metric,maps=feature_response(x,x,np.linspace(0,1,7))
        self.assertEqual(metric['cosine_delta_mean'],0.)
        self.assertEqual(metric['relative_l2_global'],0.)
        np.testing.assert_array_equal(maps['cosine_delta'],0)

    def test_fractional_in_out_boundary_and_grid_layout(self):
        base=torch.eye(4);changed=base.clone();changed[0]=base[1]
        metric,_=feature_response(changed,base,np.array([1.,.5,0.,0.]))
        self.assertGreater(metric['in_roi_cosine_delta'],0.)
        self.assertEqual(metric['outside_roi_cosine_delta'],0.)
        with self.assertRaises(ValueError):feature_response(changed,base,np.ones(3))

    def test_decomposition_reconstructs_value_plus_routing_change(self):
        torch.manual_seed(4);p0=torch.randn(1,64,196).softmax(-1);p=torch.randn_like(p0).softmax(-1)
        v0=torch.randn(1,196,8);v=v0+.1*torch.randn_like(v0)
        value=compression_decomposition(p,v,p0,v0)
        self.assertLess(value['identity_max_abs_roundoff'],1e-6)
        self.assertGreater(value['P_only_relative_norm'],0.)
        self.assertGreater(value['V_only_relative_norm'],0.)

    def test_routing_JS_zero_and_topk_overlap_one_for_same_input(self):
        r=torch.randn(1,64,196);p=r.softmax(-1)
        result=routing_response(r,p,r,p,np.ones(196)*.25)
        self.assertEqual(result['routing_JS_per_query'],0.)
        self.assertEqual(result['routing_top10_overlap'],1.)
        self.assertEqual(result['logits_top1_change_fraction'],0.)

    def test_pixel_energy_distinguishes_GT_and_actual_overlay_mask(self):
        rgb=np.zeros((3,4,3),np.uint8);changed=rgb.copy();changed[0,0]=255
        mask=np.zeros((3,4));mask[0,0]=1
        value=pixel_response(rgb,changed,mask,mask)
        self.assertEqual(value['L1_in_GT'],1.);self.assertEqual(value['L1_outside_overlay'],0.)

    def test_hook_captures_without_replacing_output(self):
        tower=nn.Linear(4,3);projector=nn.Linear(3,2)
        model=SimpleNamespace(get_vision_tower=lambda:tower,get_model=lambda:SimpleNamespace(mm_projector=projector))
        runtime=SimpleNamespace(model=model);x=torch.randn(2,4)
        expected=projector(tower(x))
        with FeatureCapture(runtime) as capture:actual=projector(tower(x))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(len(tower._forward_hooks),0)
        self.assertEqual(capture.values['encoder_patch_features'].shape,(2,3))


class SourceIsolation(unittest.TestCase):
    def row(self,source,digest='a',path='/tmp/one.mp4'):
        return {'source_id':source,'media_sha256':digest,'media_path':path}

    def test_identity_source_clip_and_youtube_alias(self):
        self.assertEqual(identity('youtube:a1234567890'),('a1234567890',None))
        self.assertEqual(identity('a1234567890_000042'),('a1234567890',42))
        self.assertIsNone(identity('a1234567890_extra_bad'))

    def test_seen_ids_and_media_hashes_forbidden_in_fresh(self):
        r=self.row('a1234567890')
        for ids,hashes in [({'a1234567890'},set()),(set(),{'a'})]:
            with self.assertRaises(ValueError):validate_pools({'fresh_holdout':[r]},ids,hashes)

    def test_same_source_or_byte_alias_cannot_cross_pools(self):
        a=self.row('a1234567890');b=self.row('b1234567890',path='/tmp/two.mp4')
        with self.assertRaises(ValueError):validate_pools({'fresh_dev':[a],'fresh_holdout':[b]},set(),set())
        with self.assertRaises(ValueError):validate_pools({'fresh_dev':[a,a]},set(),set())

    def test_fresh_cohorts_valid_if_all_three_identities_disjoint(self):
        r=validate_pools({'fresh_dev':[self.row('a')],'fresh_holdout':[self.row('b','b','/tmp/two.mp4')]},set(),set())
        self.assertEqual(r['sources'],2);self.assertFalse(r['cross_pool_leakage'])

    def test_holdout_refuses_prelock_access(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):holdout_guard(Path(temp),Path(temp)/'manifest.jsonl')

    def test_history_extracts_prefixed_and_explicit_source_lists(self):
        seen=defaultdict(set);hashes=defaultdict(set)
        collect_evidence({'source_video_id':'youtube:a1234567890','media_sha256':'digest',
                          'source_ids':['b1234567890']},'file.json',seen,hashes)
        self.assertEqual(set(seen),{'a1234567890','b1234567890'})
        self.assertEqual(hashes['digest'],{'a1234567890'})

    def test_proportional_sampling_is_fixed_without_outcome_fields(self):
        rows=[{'source_id':str(i),'stratum':'small' if i<30 else 'large'} for i in range(100)]
        selected=proportional_sample(rows,20,'test')
        self.assertEqual(selected,proportional_sample(list(reversed(rows)),20,'test'))
        self.assertEqual(sum(r['stratum']=='small' for r in selected),6)
        self.assertEqual(len({r['source_id'] for r in selected}),20)

    def test_metric_schema_keeps_proxy_count_accuracy_and_teacher_separate(self):
        method={'features':{'pixel':{'L1':.1}},'spatial':{'mass':.2,'excess_mass':.1,'lift':2.},
                'category':{'correct':True,'gold_nll':.3},'teacher':{'text':'A','option_logits':dict(zip('ABCD',[2.,1.,0.,-1.]))}}
        row={'methods':{'rgb_clean':method},'teacher_full196':method['teacher']}
        value=summarize([row],['rgb_clean'])['rgb_clean']
        self.assertEqual(value['category_proxy_correct_count'],1)
        self.assertEqual(value['category_proxy_accuracy']['mean'],1.)
        self.assertEqual(value['teacher_agreement']['mean'],1.)
        self.assertEqual(value['spatial']['mass']['mean'],.2)


if __name__=='__main__':unittest.main()
