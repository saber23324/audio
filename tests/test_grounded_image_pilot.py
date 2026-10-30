import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from streamingtom.audio_bottleneck.adbt import AudioConditionedBottleneck
from utility_gate.grounded_image_pilot.audit import collect_ids
from utility_gate.grounded_image_pilot.heatmaps import box_map, outline_map, require_map_provenance, sham_map, valid_boxes
from utility_gate.grounded_image_pilot.image_fusion import overlay
from utility_gate.grounded_image_pilot.proxy_eval import category_task, teacher_task
from utility_gate.grounded_image_pilot.routing_bias import resample
from utility_gate.grounding.spatial_mapping import ImageTransform, pooled_occupancy


class GroundedImages(unittest.TestCase):
    def test_fractional_multi_box_union_preserves_area(self):
        a=box_map([[.1,.2,.6,.7],[.4,.3,.9,.8]],20,30)
        self.assertAlmostEqual(float(a.sum()),(.25+.25-.2*.4)*600,places=4)
        np.testing.assert_array_equal(a,box_map([[.1,.2,.6,.7],[.4,.3,.9,.8],[.1,.2,.6,.7]],20,30))

    def test_normalized_xyxy_no_silent_clip_or_degenerate_boxes(self):
        for a in ([],[[0,0,0,1]],[[.8,0,.2,1]],[[-.1,0,1,1]],[[0,0,1.01,1]],[[0,0,float('nan'),1]]):
            with self.assertRaises(ValueError):valid_boxes(a)

    def test_zero_alpha_and_zero_map_exact_RGB(self):
        rgb=np.random.default_rng(4).integers(0,256,(18,23,3),dtype=np.uint8)
        h=box_map([[.2,.1,.6,.7]],18,23)
        np.testing.assert_array_equal(overlay(rgb,h,0),rgb)
        np.testing.assert_array_equal(overlay(rgb,np.zeros_like(h)),rgb)
        result=overlay(rgb,h)
        np.testing.assert_array_equal(result[h==0],rgb[h==0])
        self.assertEqual(result.dtype,np.uint8)

    def test_invalid_alpha_map_and_color_rejected(self):
        rgb=np.zeros((3,4,3),np.uint8)
        for h,alpha,color in [(np.ones((3,4))*1.1,.2,[255,0,0]),(np.zeros((3,4)),-1,[0,0,0]),
                              (np.full((3,4),np.nan),.2,[0,0,0]),(np.ones((3,4)),.2,[256,0,0])]:
            with self.assertRaises(ValueError):overlay(rgb,h,alpha,color)

    def test_sham_is_seeded_shared_translation_with_exact_intensity(self):
        h=box_map([[.2,.2,.45,.5],[.3,.4,.55,.65]],80,160)
        sham,meta=sham_map(h,'source_A')
        repeat,other=sham_map(h,'source_A')
        np.testing.assert_array_equal(sham,repeat);self.assertEqual(meta,other)
        np.testing.assert_array_equal(np.sort(h.ravel()),np.sort(sham.ravel()))
        np.testing.assert_array_equal(sham,np.roll(h,(meta['dy_pixels'],meta['dx_pixels']),(0,1)))

    def test_full_frame_sham_may_be_degenerate_without_cherry_picking(self):
        h=box_map([[0,0,1,1]],10,20);s,m=sham_map(h,'full')
        self.assertTrue(m['zero_translation']);np.testing.assert_array_equal(s,h)

    def test_outline_has_no_interior_fill(self):
        h=outline_map([[.1,.1,.9,.9]],100,100)
        self.assertGreater(h.sum(),0);self.assertEqual(h[50,50],0)

    def test_exact_grid_excludes_unconsumed_border(self):
        w=pooled_occupancy([[378/384,0,1,1]],ImageTransform(384,384))
        np.testing.assert_array_equal(w,0)

    def test_gt_is_forbidden_in_predicted_provenance(self):
        with self.assertRaises(ValueError):require_map_provenance({'map_origin':'GT_BOX_RENDERED'},True)
        with self.assertRaises(ValueError):require_map_provenance({'map_origin':'UNKNOWN'},True)

    def test_predicted_requires_causal_gt_free_model_provenance(self):
        value={'map_origin':'LEGACY_ROUTING_PRED','uses_test_annotation_as_input':False,
               'source_model':'original_frozen_router','checkpoint_sha256':'abc','audio_window':[3,5],'frame_time':5}
        require_map_provenance(value,True)
        for change in ({'audio_window':[5,7]},{'audio_window':[0,5]},{'uses_test_annotation_as_input':True},{'checkpoint_sha256':''}):
            with self.assertRaises(ValueError):require_map_provenance({**value,**change},True)


class GroundedRouting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.router=AudioConditionedBottleneck(8,64,16,2,stage=1,value_mode='native',latent_norm='none',
                                              architecture='phase4',attention_temperature=.05).eval().requires_grad_(False)
        self.visual=torch.randn(2,196,8);self.prior=torch.cat([torch.ones(40),torch.zeros(156)])

    def test_zero_bias_exact_original_P_Z_batch_and_frozen(self):
        z,p=self.router(self.visual,None,None);zero,zp=resample(self.router,self.visual,self.prior,0)
        self.assertTrue(torch.equal(z,zero) and torch.equal(p,zp))
        self.assertFalse(any(x.requires_grad for x in self.router.parameters()))

    def test_positive_bias_changes_only_P_native_values_no_extra_tokens(self):
        z,p=resample(self.router,self.visual,self.prior,.05)
        base_z,base_p=self.router(self.visual,None,None)
        self.assertEqual(z.shape,(2,64,8));self.assertTrue(torch.equal(z,p@self.visual))
        self.assertTrue((p[:,:,:40].sum(-1)>base_p[:,:,:40].sum(-1)).all())
        self.assertFalse(torch.equal(z,base_z))

    def test_uniform_prior_leaves_original_logits_unchanged(self):
        z,p=self.router(self.visual,None,None);nz,np_=resample(self.router,self.visual,torch.ones(196),.05)
        self.assertTrue(torch.equal(z,nz) and torch.equal(p,np_))

    def test_wrong_stage_or_layout_and_nan_fail(self):
        with self.assertRaises(ValueError):resample(self.router,self.visual,torch.ones(100),.05)
        with self.assertRaises(ValueError):resample(self.router,self.visual,torch.full((196,),float('nan')),.05)
        self.router.stage=3
        with self.assertRaises(ValueError):resample(self.router,self.visual,self.prior,.05)


class ProxyValidity(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).parents[1]/'utility_gate/grounded_image_pilot/configs/pilot.yaml').read_text())

    def test_category_mapping_options_fixed_and_unknown_not_fabricated(self):
        a=category_task('dog barking','some_source',self.config)
        self.assertEqual(a,category_task('dog barking','some_source',self.config))
        self.assertEqual(a['options']['ABCD'.index(a['answer'])].split('. ',1)[1],'dog')
        self.assertEqual(len(set(a['options'])),4)
        self.assertFalse(category_task('splashing water','some_source',self.config)['applicable'])

    def test_teacher_question_needs_no_GT_class_or_bbox(self):
        task=teacher_task(self.config)
        self.assertNotIn('answer',task);self.assertNotIn('bbox',task);self.assertNotIn('class',task)
        self.assertIn('not ground truth',task['task'])

    def test_historical_sources_include_ids_clips_and_training_lists(self):
        ids=set();collect_ids({'source_video_id':'a1234567890','nested':[{'clip_id':'b1234567890_000030'}],
                               'train':['c1234567890'],'sha256':'d'*64},ids)
        self.assertEqual(ids,{'a1234567890','b1234567890','c1234567890'})


if __name__=='__main__':unittest.main()
