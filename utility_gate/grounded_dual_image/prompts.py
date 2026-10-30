"""Fixed generic system prompts. No benchmark labels or sound class names."""
from __future__ import annotations
import copy
from .runtime import S0,S1

S2='''You may receive two views for the same video moment.
View 1 is the original, unmodified RGB frame.
View 2 is a derived view of that same frame with a visual highlight that
may indicate a region associated with an observed sound event.
The highlight is only a hint and may be incorrect. Use the original view
for object identity, color, counts, and scene context. Consult the second
view when spatial focus may help. Do not treat the highlighted region as
a guaranteed sound source or as the answer to the question.
Reason from the available visual evidence and respond only in the required
answer format.'''
S3='''Two images may describe the same instant from two visual renderings.
They are not two consecutive time steps. Use both as visual evidence,
prioritizing the original scene content. Follow the question and return
only the required answer format.'''
S2_SWAPPED=S2.replace('View 1 is the original','View X is the original').replace('View 2 is a derived','View 1 is a derived').replace('View X is the original','View 2 is the original')
PROMPTS={'S0':S0,'S1':S1,'S2':S2,'S3':S3,'S2_swapped':S2_SWAPPED}

def native_prompt(tokenizer,question,timestamps,system='S2',swapped=False,single=False):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
    if not str(question).strip():raise ValueError('empty question')
    if swapped and system!='S2_swapped':raise ValueError('swapped image roles require swapped prompt')
    lines=[]
    for t in timestamps:
        lines.append(f'At video time {float(t):.3f} seconds'+(':' if single else ' (two views of the same instant):'))
        if single:lines.append(DEFAULT_IMAGE_TOKEN)
        else:
            roles=('derived rendering','original RGB') if swapped else ('original RGB','derived rendering')
            lines.extend(f'View {i+1} ({role}): {DEFAULT_IMAGE_TOKEN}' for i,role in enumerate(roles))
    lines.append('Answer the following question based on the video. Respond with only a short answer.\n'+question)
    conv=copy.deepcopy(conv_templates['qwen_1_5']);conv.system='<|im_start|>system\n'+PROMPTS[system]
    conv.append_message(conv.roles[0],'\n'.join(lines));conv.append_message(conv.roles[1],None)
    return tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
