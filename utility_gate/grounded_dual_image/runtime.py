"""New open-answer adapter over the unchanged frozen AudioRouter runtime."""
from __future__ import annotations
import copy, time
import numpy as np
import torch
import torch.nn.functional as F
from utility_gate.runtime import FrozenRuntime, clocked, synchronize
from .common import ROOT, REPO, write_json
from .data import normalize

S0='You are a helpful assistant.'
S1=('You are a visual question-answering assistant.\n'
    'Use the provided visual evidence to answer the question accurately.\n'
    'Do not invent details that are not supported by the images.\n'
    'Give only the answer in the format required by this benchmark.')

def prompt(tokenizer, question, system=S0):
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    if not str(question).strip():raise ValueError('empty question')
    conv=copy.deepcopy(conv_templates['qwen_1_5'])
    conv.system='<|im_start|>system\n'+system
    # MUSIC is open answer, not a fabricated ABCD classification task.
    text='Answer the following question based on the video. Respond with only a short answer.\n'+question
    conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+text)
    conv.append_message(conv.roles[1],None)
    return tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')

def insert_views(model, ids, views):
    from llava.constants import IMAGE_TOKEN_INDEX
    ids=ids.reshape(-1);positions=torch.where(ids==IMAGE_TOKEN_INDEX)[0].tolist()
    if len(positions)!=len(views):raise ValueError('image marker / actual view count mismatch')
    embed=model.get_input_embeddings();device,dtype=embed.weight.device,embed.weight.dtype
    chunks=[];start=0
    for position,view in zip(positions,views):
        chunks.extend((embed(ids[start:position].to(device)),view.reshape(-1,view.shape[-1]).to(device,dtype)))
        start=position+1
    chunks.append(embed(ids[start:].to(device)))
    return torch.cat(chunks,0)[None]

class QARuntime(FrozenRuntime):
    @torch.inference_mode()
    def interface_probe(self):
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        images=[Image.fromarray(np.full((384,384,3),v,dtype=np.uint8)) for v in (40,180)]
        tensors=[self.processor.preprocess(im,return_tensors='pt')['pixel_values'].to(self.device,dtype=torch.float16) for im in images]
        report={}
        for n in (1,2):
            ids=torch.tensor([[10]+[IMAGE_TOKEN_INDEX]*n+[11]],device=self.device)
            values=self.model.prepare_inputs_labels_for_multimodal(ids,None,None,None,None,
                tensors[:n],modalities=['image']*n,image_sizes=[im.size for im in images[:n]])
            embeddings=values[4]
            report[str(n)]={'processor_images':n,'processor_shapes':[list(t.shape) for t in tensors[:n]],
                'native_modalities':['image']*n,'image_markers':n,'native_inserted_tokens':embeddings.shape[1]-2,
                'native_embedding_shape':list(embeddings.shape)}
        if report['2']['native_inserted_tokens']!=2*report['1']['native_inserted_tokens']:raise AssertionError('native image token count')
        write_json(ROOT/'audit/native_interface_probe.json',report)
        return report

    @torch.inference_mode()
    def answer(self, ids, views, gold_text, max_new_tokens=16, prepared=None, mcqa=False, compute_nll=True):
        # Only the evaluation teacher pass receives gold; greedy generation has
        # already completed and has no gold argument in the native decoder.
        synchronize();start=time.perf_counter()
        inputs=insert_views(self.model,ids,views) if prepared is None else prepared
        device=inputs.device;length=inputs.shape[1]
        attention=torch.ones((1,length),device=device,dtype=torch.long)
        positions=torch.arange(length,device=device)[None]
        synchronize();prefill=time.perf_counter()
        output=self.model.model(inputs_embeds=inputs,attention_mask=attention,position_ids=positions,use_cache=True,return_dict=True)
        logits=self.model.lm_head(output.last_hidden_state[:,-1]).float()[0]
        synchronize();prefill_s=time.perf_counter()-prefill
        generated=[int(logits.argmax())];ttft=time.perf_counter()-start
        first_logits=logits
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos if eos is not None else self.tokenizer.eos_token_id])
        cache=output.past_key_values
        for step in range(1,max_new_tokens):
            if generated[-1] in eos:break
            output=self.model.model(input_ids=torch.tensor([[generated[-1]]],device=device),
                attention_mask=torch.ones((1,length+step),device=device,dtype=torch.long),
                position_ids=torch.tensor([[length+step-1]],device=device),past_key_values=cache,use_cache=True,return_dict=True)
            cache=output.past_key_values
            logits=self.model.lm_head(output.last_hidden_state[:,-1]).float()[0]
            generated.append(int(logits.argmax()))
        synchronize();generation_s=time.perf_counter()-start
        text=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        del cache,output
        generation_peak=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if not compute_nll:
            return {'text':text,'generated_ids':generated,'visual_tokens':sum(len(v.reshape(-1,v.shape[-1])) for v in views),
                'total_input_tokens':length,'llm_prefill_s':prefill_s,'llm_ttft_s':ttft,'llm_total_s':generation_s,
                'generation_peak_allocated_bytes':generation_peak,'nll_evaluated':False,'max_new_tokens':max_new_tokens}
        target=self.tokenizer.encode(str(gold_text).strip(),add_special_tokens=False)
        if not target:raise ValueError('empty gold answer')
        # Explicit custom sequence NLL, not official MUSIC classifier NLL.
        # Re-prefill to avoid reusing a greedy-mutated cache in teacher forcing.
        teacher_start=time.perf_counter()
        if len(target)==1:
            token_nll=[float(-first_logits.log_softmax(0)[target[0]])]
        else:
            added=self.model.get_input_embeddings()(torch.tensor([target[:-1]],device=device))
            teacher=torch.cat((inputs,added),1)
            result=self.model.model(inputs_embeds=teacher,attention_mask=torch.ones(teacher.shape[:2],device=device,dtype=torch.long),
                position_ids=torch.arange(teacher.shape[1],device=device)[None],use_cache=False,return_dict=True)
            gold_logits=self.model.lm_head(result.last_hidden_state[:,length-1:]).float()[0]
            token_nll=F.cross_entropy(gold_logits,torch.tensor(target,device=device),reduction='none').cpu().tolist()
        synchronize()
        result={'text':text,'generated_ids':generated,'prediction':normalize(text),'correct':normalize(text)==normalize(gold_text),
            'strict_normalized_correct':normalize(text,False)==normalize(gold_text,False),'gold_nll':sum(token_nll),
            'gold_nll_per_token':sum(token_nll)/len(target),'gold_token_ids':target,'gold_token_nll':token_nll,
            'nll_protocol':'custom teacher-forced canonical official answer string, full vocabulary, sum over answer tokens, no EOS; not official classification NLL',
            'visual_tokens':sum(len(v.reshape(-1,v.shape[-1])) for v in views),'total_input_tokens':length,
            'llm_prefill_s':prefill_s,'llm_ttft_s':ttft,'llm_total_s':generation_s,'nll_evaluation_extra_s':time.perf_counter()-teacher_start,
            'max_new_tokens':max_new_tokens,'stopped_eos':generated[-1] in eos,
            'generation_peak_allocated_bytes':generation_peak,
            'evaluation_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        if mcqa:
            option_ids=[self.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD']
            if any(len(v)!=1 for v in option_ids) or gold_text not in 'ABCD':raise ValueError('MCQA requires native single-letter labels')
            option_logits=first_logits[torch.tensor([v[0] for v in option_ids],device=device)]
            result.update(gold_nll_full_vocab=result['gold_nll'],gold_nll=float(-option_logits.log_softmax(0)['ABCD'.index(gold_text)]),
                option_logits=dict(zip('ABCD',option_logits.cpu().tolist())),
                correct=text==gold_text,strict_normalized_correct=text==gold_text,prediction=text if text in ('A','B','C','D') else None,
                nll_protocol='single gold option letter; first-token log-softmax over A/B/C/D; full-vocabulary NLL retained')
        return result

    @torch.inference_mode()
    def encode_music(self,row):
        from relevance_gate.media import frame_at_or_before
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        # Four uniform, question-independent full-observation timestamps. Audio
        # for each early frame is truncated before resampling at that frame.
        timestamps=np.asarray([.125,.375,.625,.875],dtype=np.float32)*float(row['duration_s'])
        def decode():
            values=[frame_at_or_before(row['video_path'],float(t)) for t in timestamps]
            return np.stack([v[0] for v in values]),[v[1] for v in values]
        (frames,pts),decode_s=clocked(decode)
        visual,vision_s=clocked(lambda:self.extractor.encode_visual(self.model,self.processor,frames,4).float())
        (windows,valid,reasons,rms),audio_decode_s=clocked(lambda:self.causal_windows(row['video_path'],timestamps))
        audio,audio_s=clocked(lambda:self.audio_encoder.encode_windows(windows,self.device).to(self.device).float())
        null,null_s=clocked(lambda:self.audio_encoder.encode_windows(torch.zeros_like(windows),self.device).to(self.device).float())
        times=torch.tensor(timestamps,device=self.device)
        (zv,pv),v_s=clocked(lambda:self.vision_router(visual,None,times))
        (za,pa),a_s=clocked(lambda:self.audio_router(visual,audio,times))
        (zn,pn),n_s=clocked(lambda:self.audio_router(visual,null,times))
        if not torch.equal(zv,pv@visual) or not torch.equal(za,pa@visual):raise AssertionError('native values changed')
        return {'frames':frames,'timestamps':times.cpu(),'frame_pts_s':pts,'visual':visual.cpu(),
            'V64':zv.cpu(),'V196':self.full_vision(visual).cpu(),'A64':za.cpu(),'A64-null':zn.cpu(),
            'vision_p':pv.cpu(),'audio_p':pa.cpu(),'audio_windows':windows,'audio_valid':valid,'audio_reasons':reasons,'audio_rms':rms,
            'timing':{'video_decode_s':decode_s,'visual_encoder_s':vision_s,'audio_decode_s':audio_decode_s,'audio_encoder_s':audio_s,
                'null_audio_encoder_s':null_s,'vision_resampling_s':v_s,'audio_resampling_s':a_s,'null_resampling_s':n_s,
                'all_baseline_encoding_s':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}}

    @torch.inference_mode()
    def regressions(self):
        values={}
        for name,router in [('m0-vision-20261022',self.vision_router),('m0-gate-disabled-final-20261022',self.audio_router)]:
            f=torch.load(REPO/'artifacts/relevance_gate'/name/'fixture.pt',map_location='cpu',weights_only=True)
            visual=torch.cat([c['visual'] for c in f['chunks']]).to(self.device)
            audio=f['audio_info']['embeddings'].to(self.device) if router.stage==3 else None
            z,p=router(visual,audio,f['timestamps'].to(self.device))
            exact_z=torch.equal(z.cpu(),torch.cat([c['latents'] for c in f['chunks']]))
            exact_p=torch.equal(p.cpu(),torch.cat([c['attention'] for c in f['chunks']]))
            generation=super().generate(f['prompt_ids'],z)
            expected=f['generated_ids'].reshape(-1).tolist()
            if not(exact_z and exact_p and generation['generated_ids']==expected):raise AssertionError(name)
            values[name]={'P_exact':True,'Z_exact':True,'generated_ids':expected,'generated_exact':True}
        if any(p.requires_grad for p in self.model.parameters()):raise AssertionError('unfrozen backbone')
        write_json(ROOT/'audit/legacy_gpu_regressions.json',values)
        return values
