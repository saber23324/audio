import copy
import unittest

import numpy as np

from utility_gate.grounded_image_pilot.report import eligible, judgment, mean_ci, pair, summarize_methods, teacher_measures


def generation(letter='A'):
    return {'text':letter,'option_logits':{'A':2.,'B':0.,'C':-1.,'D':-2.}}


def row(source,a,b):
    methods={}
    for name,correct in [('cue',a),('base',b)]:
        methods[name]={'spatial':{'mass':float(correct),'excess_mass':float(correct)-.2,
                         'lift':float(correct)*5,'pointing':correct},
                       'teacher':generation('A' if correct else 'B'),
                       'category':{'correct':correct,'gold_nll':.2 if correct else 1.5}}
    return {'source_id':source,'methods':methods,'teacher_full196':generation()}


class PairedReport(unittest.TestCase):
    def test_summary_retains_accuracy_interval_separately_from_correct_count(self):
        r=row('one',True,False)
        value=r['methods'].pop('cue');r['methods']={'rgb_clean':value}
        value['category']['parser_valid']=True
        value['routing_diagnostics']={'mean_row_cosine':.5}
        value['timing']={'visual_encoder_s':.01}
        value['teacher'].update(llm_prefill_s=.01,llm_ttft_s=.02,llm_total_s=.03)
        result=summarize_methods([r])['rgb_clean']['category_proxy']
        self.assertEqual(result['correct_count'],1)
        self.assertEqual(result['correct']['mean'],1.)
        self.assertEqual(result['correct']['ci95'],[1.,1.])

    def test_rescue_harm_quadrants_and_endpoints(self):
        rows=[row('both',True,True),row('rescue',True,False),row('harm',False,True),row('neither',False,False)]
        q=pair(rows,'cue','base')['category_proxy']
        self.assertEqual([q[k] for k in ('both_correct','cue_only_win','vision_only_win','both_wrong')],[1,1,1,1])
        self.assertEqual(q['oracle_endpoints_correct'],3);self.assertEqual(q['correct']['mean'],0.)

    def test_pair_uses_common_denominator_and_masks_absent_proxy(self):
        rows=[row('a',True,False),row('b',False,True),row('c',True,False)]
        del rows[1]['methods']['cue'];del rows[2]['methods']['cue']['category']
        result=pair(rows,'cue','base')
        self.assertEqual(result['n'],2);self.assertEqual(result['category_proxy']['n'],1)
        self.assertEqual(result['category_proxy']['correct']['mean'],1.)

    def test_duplicate_sources_do_not_become_independent_observations(self):
        r=row('same',True,True)
        with self.assertRaises(ValueError):eligible([r,copy.deepcopy(r)],'cue')

    def test_bootstrap_finite_and_constant_zero(self):
        result=mean_ci([0.,0.,0.])
        self.assertEqual(result['mean'],0.);self.assertEqual(result['ci95'],[0.,0.])
        self.assertIsNone(mean_ci([])['mean'])
        with self.assertRaises(ValueError):mean_ci([float('nan')])

    def test_teacher_is_reference_and_invalid_generation_not_a_fake_match(self):
        r=row('s',True,True)
        self.assertAlmostEqual(teacher_measures(r,'cue')['kl_teacher_to_student'],0.)
        r['teacher_full196']['text']='ambiguous'
        self.assertIsNone(teacher_measures(r,'cue')['generated_agreement'])
        r['methods']['cue']['teacher']['option_logits']={'A':-2.,'B':2.,'C':0.,'D':0.}
        self.assertGreater(teacher_measures(r,'cue')['kl_teacher_to_student'],0.)

    def test_oracle_go_never_implies_deployment_go(self):
        values={'category_proxy':{'correct':{'mean':.1},'gold_nll':{'ci95':[-.3,-.1]}},
                'spatial':{'mass':{'ci95':[.1,.2]}},'teacher_fidelity':{'kl_teacher_to_student':{'ci95':[-.3,-.1]}}}
        result=judgment({'gt_overlay_minus_rgb_clean':values,'gt_overlay_minus_sham_overlay':values})
        self.assertTrue(result['GO_1_oracle_spatial_cue_potential']);self.assertFalse(result['GO_2_deployable_audio_cue'])
        self.assertTrue(result['NO_GO_3'])


if __name__=='__main__':unittest.main()
