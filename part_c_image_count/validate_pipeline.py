"""CPU-only official processor pilot. No weights, forward pass, or training.

Run from any directory. Output directories must not already exist.
"""
from pathlib import Path
import argparse, csv, hashlib, inspect, io, json, platform, sys, time
import PIL
import numpy as np
from PIL import Image, ImageDraw
import transformers
from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2TokenizerFast

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--assets-dir', type=Path, default=ROOT / 'assets', help='Pinned small tokenizer/config assets, never model weights')
ap.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / 'pilot', help='A new run directory; never overwritten')
ap.add_argument('--samples-per-count', type=int, default=8, help='Aligned samples per N; 8 reproduces the original pilot')
ap.add_argument('--save-images', action='store_true', help='Optionally retain generated PNGs; seeds and hashes are always recorded')
args = ap.parse_args()
if args.samples_per_count < 1:
    ap.error('--samples-per-count must be positive')
MODEL = args.assets_dir.resolve()
expected = json.loads((ROOT / 'assets_manifest.json').read_text())
for asset in expected['files']:
    path = MODEL / asset['name']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
        raise ValueError(f"Missing or unexpected asset: {asset['name']}. Run download_reference.py first.")
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
DATA = OUT / 'images'
if args.save_images:
    DATA.mkdir(exist_ok=False)
SEED = 55402026
SHAPES = {1:(672,672), 2:(448,504), 3:(336,448), 4:(336,336), 8:(224,252)} # H,W
tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL, local_files_only=True)
rows, manifest = [], []
config = dict(seed=SEED, samples_per_count=args.samples_per_count, counts=list(SHAPES), target=576,
              aligned_shapes_hw=SHAPES, min_pixels=3136, max_pixels_rule='576/N * 28^2',
              allocation='equal visual tokens per image, constructed aspect ratios',
              mixtures={'A2-Low':{'1':1.0}, 'A2-Med':{'1':1/3,'2':1/3,'3':1/3},
                        'A2-High':{'1':.25,'2':.25,'4':.25,'8':.25}},
              mixture_status='temporary balanced pilot, reuse the recorded N-strata',
              scope='official image processor and tokenizer only; no model weights or forward')
(OUT/'config.json').write_text(json.dumps(config,indent=2))

def synthetic(h,w,seed):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:h,:w]
    color=rng.integers(20,220,3)
    arr=np.stack([(x//7+color[0])%256,(y//9+color[1])%256,((x+y)//11+color[2])%256],axis=-1).astype('uint8')
    im=Image.fromarray(arr)
    ImageDraw.Draw(im).text((12,12),f'unique seed {seed}',fill='white')
    return im

def run(n,idx,kind):
    h,w = SHAPES[n] if kind=='aligned' else (672,672)
    # Original dimensions are exactly twice the requested processor dimensions.
    images=[]; entries=[]
    sid=f'{kind}_N{n}_{idx:02d}'
    for j in range(n):
        seed=SEED+n*10000+idx*100+j
        im=synthetic(2*h,2*w,seed)
        path=DATA/f'{sid}_image{j:02d}.png'
        png = io.BytesIO()
        im.save(png, format='PNG')
        if args.save_images:
            path.write_bytes(png.getvalue())
        entries.append(dict(image_id=f'{sid}:{j}',path=f'images/{path.name}' if args.save_images else None,
                            seed=seed,original_hw=[2*h,2*w],sha256=hashlib.sha256(png.getvalue()).hexdigest()))
        images.append(im)
    # These are official, implemented parameters. No token truncation or crop.
    ip=Qwen2VLImageProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=3136,max_pixels=(576//n)*784)
    proc=Qwen2VLProcessor(image_processor=ip,tokenizer=tokenizer,chat_template=tokenizer.chat_template)
    prompt=proc.apply_chat_template([{'role':'user','content':
        [{'type':'image'} for _ in images]+[{'type':'text','text':'Describe these images.'}]}],tokenize=False,add_generation_prompt=True)
    start=time.perf_counter()
    batch=proc(text=[prompt],images=images,return_tensors='np',padding=False,truncation=False)
    cpu_ms=(time.perf_counter()-start)*1000
    grid=batch['image_grid_thw']; pixels=batch['pixel_values']
    patch_counts=np.prod(grid,axis=1).astype(int)
    # Prediction based on real patch-grid output and official 2x2 spatial merge.
    expected_llm=(patch_counts//ip.merge_size**2).astype(int)
    offsets=np.r_[0,np.cumsum(patch_counts)]
    slice_matches=[]; perturb_matches=[]
    for j,im in enumerate(images):
        single=ip(images=[im],return_tensors='np')
        slice_matches.append(bool(np.array_equal(single['pixel_values'],pixels[offsets[j]:offsets[j+1]])))
        # Perturb every image in turn; its own tensor segment must change and all
        # other segments must remain byte-for-byte identical.
        changed=list(images)
        changed[j]=Image.fromarray(255-np.asarray(im))
        alt=ip(images=changed,return_tensors='np')['pixel_values']
        segment_changes=[not np.array_equal(alt[offsets[k]:offsets[k+1]],pixels[offsets[k]:offsets[k+1]]) for k in range(n)]
        perturb_matches.append(segment_changes==[k==j for k in range(n)])
    ids=batch['input_ids'][0]
    image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vs=tokenizer.convert_tokens_to_ids('<|vision_start|>'); ve=tokenizer.convert_tokens_to_ids('<|vision_end|>')
    slot_count=int(np.sum(ids==image_id))
    specials=int(np.sum(ids==vs)+np.sum(ids==ve))
    padding=int(np.sum(batch['attention_mask']==0))
    all_images=bool(len(grid)==n and all(slice_matches) and all(perturb_matches))
    assert all_images and slot_count==int(sum(expected_llm)) and specials==2*n and padding==0
    row=dict(sample_id=sid,kind=kind,image_count=n,source='synthetic pipeline-only',
        original_hw=[e['original_hw'] for e in entries],processed_hw=[[int(g[1])*14,int(g[2])*14] for g in grid],
        grid_thw=grid.tolist(),image_blocks=n,spatial_crops=0,pixel_values_shape=list(pixels.shape),
        grid_shape=list(grid.shape),encoder_input_patch_rows_per_image=patch_counts.tolist(),
        encoder_input_patch_rows_total=int(sum(patch_counts)),
        llm_visual_tokens_derived_per_image=expected_llm.tolist(),llm_visual_tokens_derived_total=int(sum(expected_llm)),
        llm_embeddings_measured='NOT MEASURED',encoder_forward_tokens='NOT MEASURED',
        expanded_image_token_slots_observed=slot_count,vision_special_markers=specials,padding_tokens=padding,
        text_and_other_special_tokens=int(len(ids)-slot_count-specials),sequence_length=int(len(ids)),
        target_visual_tokens=576,budget_delta=int(sum(expected_llm))-576,
        image_tensor_slices_match=all(slice_matches),each_image_perturbation_pass=all(perturb_matches),
        all_images_retained_in_processor=all_images,actual_llm_consumption='NOT MEASURED',
        original_pixels=sum(e['original_hw'][0]*e['original_hw'][1] for e in entries),
        processed_pixels=sum(int(g[1])*14*int(g[2])*14 for g in grid),
        cpu_processor_ms_uncontrolled=round(cpu_ms,3),encoder_ms='NOT MEASURED',training_step_ms='NOT MEASURED',
        status='PASS')
    rows.append(row)
    manifest.append(dict(sample_id=sid,kind=kind,image_count=n,images=entries))
    print(json.dumps({k:row[k] for k in ['sample_id','pixel_values_shape','llm_visual_tokens_derived_total','budget_delta','all_images_retained_in_processor','status']}),flush=True)

for n in SHAPES:
    for i in range(args.samples_per_count): run(n,i,'aligned')
for n in SHAPES: run(n,0,'square_stress')

# Mixed-batch tokenizer padding is measured separately from visual slots.
texts=['<|vision_start|><|image_pad|><|vision_end|>'*n+' Describe.' for n in [1,8]]
ip=Qwen2VLImageProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=3136,max_pixels=72*784)
proc=Qwen2VLProcessor(image_processor=ip,tokenizer=tokenizer)
mixed=proc(text=texts,images=[synthetic(448,504,SEED+i) for i in range(9)],return_tensors='np',padding=True,truncation=False)
mixed_result=dict(image_grid_thw=mixed['image_grid_thw'].tolist(),pixel_values_shape=list(mixed['pixel_values'].shape),
                  input_ids_shape=list(mixed['input_ids'].shape),padding_tokens_per_sample=(1-mixed['attention_mask']).sum(axis=1).tolist(),
                  image_slots_per_sample=[int(np.sum(ids==tokenizer.convert_tokens_to_ids('<|image_pad|>'))) for ids in mixed['input_ids']],
                  purpose='padding accounting check only; common 72-token per-image cap; not shared-budget experiment')
(OUT/'mixed_batch_padding.json').write_text(json.dumps(mixed_result,indent=2))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'samples.json').write_text(json.dumps(rows,indent=2))
with (OUT/'samples.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
    writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
summary=[]
for n in SHAPES:
    rr=[r for r in rows if r['kind']=='aligned' and r['image_count']==n]
    stress=next(r for r in rows if r['kind']=='square_stress' and r['image_count']==n)
    summary.append(dict(n=n,aligned_hw=rr[0]['processed_hw'][0],aligned_samples=len(rr),
                        aligned_total_tokens=rr[0]['llm_visual_tokens_derived_total'],patch_rows=rr[0]['encoder_input_patch_rows_total'],
                        preserved=sum(r['all_images_retained_in_processor'] for r in rr),square_stress_total_tokens=stress['llm_visual_tokens_derived_total']))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
versions=dict(python=sys.version,transformers=transformers.__version__,numpy=np.__version__,platform=platform.platform(),
              device='CPU, numpy processor only',warmup='none',repeats=f'{args.samples_per_count} independently constructed samples per N, not benchmark repetitions',
              pillow=PIL.__version__,model=expected['repo'],revision=expected['revision'],
              encoder_forward=False,training=False)
(OUT/'environment.json').write_text(json.dumps(versions,indent=2))
source_hashes = {}
for obj in [Qwen2VLImageProcessor, Qwen2VLProcessor]:
    src = Path(inspect.getfile(obj))
    source_hashes[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
(OUT/'processor_source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
