import json
import os
import time
import warnings
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pysam
import pyfaidx
import pybedtools
import csv
import tensorflow as tf

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dna

from borzoi_helpers import *

import pdb
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'












###################################
borzoi_downloads_dir = sys.argv[1]
borzoi_examples_data_dir = sys.argv[2]
###################################

params_file = borzoi_examples_data_dir + 'params_pred.json'
targets_file = borzoi_examples_data_dir + 'targets_gtex.txt' #Subset of targets_human.txt




pyfaidx.Faidx(borzoi_downloads_dir +'hg38/assembly/ucsc/hg38.fa')


#Model configuration

seq_len = 524288
n_reps = 1       #To use only one model replicate, set to 'n_reps = 1'. To use all four replicates, set 'n_reps = 4'.
rc = True         #Average across reverse-complement prediction

#Read model parameters

with open(params_file) as params_open :
	
	params = json.load(params_open)
	
	params_model = params['model']
	params_train = params['train']

#Remove cropping
params_model['trunk'][-2]['cropping'] = 0

#Read targets

targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

#Create local index of strand_pair (relative to sliced targets)
if rc :
	strand_pair = targets_df.strand_pair
	
	target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
	slice_pair = np.array([
		target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
	], dtype='int32')

#Initialize model ensemble

models = []
for rep_ix in range(n_reps) :

	print(rep_ix)
	
	model_file = borzoi_downloads_dir + 'saved_models/f3c' + str(rep_ix) + '/train/model0_best.h5'

	seqnn_model = seqnn.SeqNN(params_model)
	seqnn_model.restore(model_file, 0)
	seqnn_model.build_slice(target_index)
	if rc :
		seqnn_model.strand_pair.append(slice_pair)
	seqnn_model.build_ensemble(rc, [0])
	
	models.append(seqnn_model)







#Load genome fasta and gene annotations

#Initialize fasta sequence extractor
fasta_open = pysam.Fastafile(borzoi_downloads_dir + 'hg38/assembly/ucsc/hg38.fa')

#Load gene/exon annotation
gtf_file = borzoi_downloads_dir + 'hg38/genes/gencode41/gencode41_basic_nort_protein.gtf'

transcriptome = bgene.Transcriptome(gtf_file)

#Get gene span bedtool
bedt_span = transcriptome.bedtool_span()

#Load APA atlas
apa_df = pd.read_csv(borzoi_downloads_dir + 'hg38/genes/polyadb/polyadb_human_v3.csv.gz', sep='\t', compression='gzip')
apa_df = apa_df[['pas_id', 'gene', 'chrom', 'position_hg38', 'strand', 'site_num', 'num_sites', 'site_type', 'pas_type', 'total_count']]

apa_df.loc[apa_df['pas_type'] == 'NoPAS', 'pas_type'] = 'No_CSE'

#Only consider 3' UTR sites
apa_df_utr = apa_df.query("site_type == '3\\' most exon'").copy().reset_index(drop=True)

#Or intronic sites
apa_df_intron = apa_df.query("site_type == 'Intron' and pas_type != 'No_CSE'").copy().reset_index(drop=True)

print("len(apa_df_utr) = " + str(len(apa_df_utr)))
print("len(apa_df_intron) = " + str(len(apa_df_intron)))

#Load TSS atlas
tss_df = pd.read_csv(borzoi_downloads_dir + 'hg38/genes/gencode41/gencode41_basic_tss2.bed', sep='\t', names=['chrom', 'position_hg38', 'end', 'tss_id', 'feat1', 'strand'])
tss_df['gene'] = tss_df['tss_id'].apply(lambda x: x.split("/")[1] if "/" in x else x)

print("len(tss_df) = " + str(len(tss_df)))


#Get reference/alternate sequence for variant, and annotations for target gene

search_gene = 'ENSG00000187164'

center_pos = 116952944

chrom = 'chr10'
poses = [116952944]
alts = ['C']

start = center_pos - seq_len // 2
end = center_pos + seq_len // 2

load_isoforms = True

#Get exon bin range
gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]

gene = transcriptome.genes[gene_keys[0]]
gene_strand = gene.strand

if chrom is None or start is None or end is None :
	chrom = gene.chrom
	g_start, g_end = gene.span()
	mid = (g_start + g_end) // 2
	start = mid - seq_len // 2
	end = mid + seq_len // 2

#Determine output sequence start
seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

#Determine output positions of gene exons
gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False, old_version=True)

#Get sequence bedtool
seq_bedt = pybedtools.BedTool('%s %d %d' % (chrom, start, end), from_string=True)

#Get all genes (exons and strands) overlapping input window
gene_ids = sorted(list(set([overlap[3] for overlap in bedt_span.intersect(seq_bedt, wo=True) if search_gene not in overlap[3]])))
gene_slices = []
gene_strands = []
for gene_id in gene_ids :
	gene_slices.append(transcriptome.genes[gene_id].output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False, old_version=True))
	gene_strands.append(transcriptome.genes[gene_id].strand)

#Get 3' UTR pA sites for gene
apa_df_gene_utr = apa_df_utr.query("gene == '" + gene.name + "'").copy().reset_index(drop=True)[['chrom', 'gene', 'strand', 'position_hg38']]
apa_df_gene_intron = apa_df_intron.query("gene == '" + gene.name + "'").copy().reset_index(drop=True)[['chrom', 'gene', 'strand', 'position_hg38']]

#Get TSS sites for gene
tss_df_gene = tss_df.loc[tss_df['gene'].str.contains(search_gene)].copy().reset_index(drop=True)[['chrom', 'gene', 'strand', 'position_hg38']]

def _switch_transcript_id(id_str) :
	return id_str.replace("gene_id", "gene_id_orig").replace("transcript_id", "gene_id")

#Get gene isoforms
isoform_slices = None
if load_isoforms :
	gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, names=['chrom', 'havana_str', 'feature', 'start', 'end', 'feat1', 'strand', 'feat2', 'id_str'])
	gtf_df = gtf_df.loc[gtf_df['id_str'].str.contains(search_gene)].copy().reset_index(drop=True)
	gtf_df = gtf_df.loc[gtf_df['id_str'].str.contains("transcript_id")].copy().reset_index(drop=True)
	gtf_df = gtf_df.loc[gtf_df['feature'] == 'exon'].copy().reset_index(drop=True)
	
	transcript_ids = gtf_df['id_str'].apply(lambda x: x.split("transcript_id \"")[1].split("\";")[0]).unique().tolist()
	gtf_df['id_str'] = gtf_df['id_str'].apply(_switch_transcript_id)
	
	gtf_df.to_csv('borzoi_gene_isoforms.gtf', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)
	
	transcriptome_iso = bgene.Transcriptome('borzoi_gene_isoforms.gtf')
	
	isoform_slices = []
	for transcript_id in transcript_ids :
		isoform_slices.append(transcriptome_iso.genes[transcript_id].output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False, old_version=True))




#Predict for chr10_116952944_T_C
# (~6 minutes on CPU w 1 replicate; ~2 minutes on GPU)

save_figs = False
save_suffix = '_chr10_116952944_T_C'

sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)

#Induce mutation(s)
sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

for pos, alt in zip(poses, alts) :
	alt_ix = -1
	if alt == 'A' :
		alt_ix = 0
	elif alt == 'C' :
		alt_ix = 1
	elif alt == 'G' :
		alt_ix = 2
	elif alt == 'T' :
		alt_ix = 3

	sequence_one_hot_mut[pos-start-1] = 0.
	sequence_one_hot_mut[pos-start-1, alt_ix] = 1.



#Make predictions
y_wt = predict_tracks(models, sequence_one_hot_wt)
y_mut = predict_tracks(models, sequence_one_hot_mut)



pdb.set_trace()