"""Unmodified ARCD functions, loaded without its dataset/CLI side effects."""
import ast
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from repro.baseline import sha256

OFFICIAL=Path(__file__).with_name('arcd_official')

def load_official():
    origin=json.loads((OFFICIAL/'origin.json').read_text())
    for name,h in origin['files'].items():
        if sha256(OFFICIAL/name)!=h:raise ValueError('ARCD source changed: '+name)
    names={'_embedding_hook','_attention_hook','install_paper_highlighter','set_paper_highlight',
           'insert_separator','generate_bbox_mask'}
    tree=ast.parse((OFFICIAL/'evaluate_ARegion.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('Missing pinned ARCD function')
    ns={'torch':torch,'F':F,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OFFICIAL/'evaluate_ARegion.py'),'exec'),ns)
    tree=ast.parse((OFFICIAL/'guidance.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProbCFGLogitsProcessor')
    ns['LogitsProcessor']=LogitsProcessor
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(OFFICIAL/'guidance.py'),'exec'),ns)
    return ns,origin


def build_inputs(processor,example,mask,mask_value=1):
    api,_=load_official()
    prompt=processor.tokenizer.apply_chat_template([{'role':'user','content':f"<|image_1|>\n{example['question']}"}],tokenize=False,add_generation_prompt=True)
    # Same ARCD processor arguments, no --use_text Object suffix (official default).
    data=processor(text=prompt,images=[example['image']],images_mask=[mask],
        bboxes=[example['bbox']],mask_values=[mask_value],attribute=[example.get('attribute','match')],
        qs_highlighted_parts=example.get('highlights',[]),return_tensors='pt')
    spatial=api['generate_bbox_mask'](data['bbox_attention_mask'],4,4).tolist()
    ids=[];highlight=[]
    for tokens,values in zip(api['insert_separator'](data['prompt_chunks'],data['image_ids_pad']),
                            api['insert_separator'](data['highlight_attention_mask'],spatial)):
        ids.extend(tokens);highlight.extend(values)
    ids=torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    h=torch.tensor(highlight,dtype=torch.bool).flatten()
    if h.numel()!=ids.shape[1]:raise ValueError('ARCD mask/token length mismatch')
    return {'input_ids':ids.repeat(2,1),'attention_mask':(ids>-1000000).long().repeat(2,1),
            'pixel_values':data['pixel_values'].repeat(2,1,1,1,1),
            'image_sizes':data['image_sizes'].repeat(2,1)},h,prompt


def generation_kwargs(processor,api):
    # Values and flags from the pinned ARCD evaluator, including its CFG class.
    return dict(eos_token_id=processor.tokenizer.eos_token_id,max_new_tokens=256,num_beams=1,
        logits_processor=[api['ProbCFGLogitsProcessor'](guidance_scale=1.5,use_log=True)],
        output_scores=True,return_dict_in_generate=True,use_cache=True)
