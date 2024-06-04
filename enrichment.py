#!/usr/bin/python

import time, argparse, numpy as np
import pyBigWig as pbw
import pandas as pd
from typing import List
from scipy import stats
import saturation_fast as sf
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type-data', choices=['chip','cut'])	# option that takes a value
parser.add_argument('-f', '--filename', nargs='+', type=str, 
					help='path to ip then input bigwig file for chip')
parser.add_argument('-b', '--bam', nargs='+', type=str, 
					help='path to ip then input bamfile for chip')
parser.add_argument('-o', '--output', type=str, 
					help='filename of the output (Dataframe)')
args = parser.parse_args()

def prepare_data(type_data:str, filename:List[str]):
	""" """
	ip_f = pbw.open(filename[0])
	chr_len = ip_f.chroms()
	chr_len.pop('chrY', None)
	chr_len = {name:int(np.ceil(val/128)) for name, val in chr_len.items()}
	
	all_ip = np.hstack([
				ip_f.values(chr, 0, -1, numpy=True)[::128].astype(np.int16)
				for chr in tqdm(chr_len)])

	if type_data == 'chip':
		in_f = pbw.open(filename[1])
		
		all_in = np.hstack([
				in_f.values(chr, 0, -1, numpy=True)[::128].astype(np.int16) 
				for chr in tqdm(chr_len)])
		return chr_len, all_ip, all_in

	return chr_len, all_ip

def repeats_pos(repeats_df, chr_len:dict):
	pos = {}
	for chr in chr_len.keys():
		dtf = repeats_df[(repeats_df.chrom == chr)]
		pos[chr] = np.zeros(chr_len[chr], dtype=bool)
		if not dtf.empty:
			start = np.ceil(dtf.start/128)
			end = dtf.end//128
			a = (np.unique(
				np.concatenate(
					[np.arange(i,j + 1) if i < j + 1 else np.empty(shape=0) for i,j in zip(start, end)])
					) - 1).astype(np.int32)
			pos[chr][a] = True
	return np.hstack(list(pos.values()))

def reads_inrep(ip, frag_ip, input=None, frag_in=None, repeats_df=None, mask=None):
	""" """
	if repeats_df is not None:
		mask = repeats_pos(repeats_df, chr_len)
		if mask is not None:
			n = np.sum(ip[mask]) / frag_ip
			m = np.sum(input[mask]) / frag_in
			if n==np.NaN or m==np.NaN:
				print(n, m)
			if n==np.inf or m==np.inf:
				print(n, m)
			return (n, m)
		else:
			return (None, None)
	else:
		if mask is not None:
			print(np.sum(mask))
			n = np.sum(ip[mask]) / frag_ip
			return n
		else:
			return None

def enrich(type_data:str, 
		   mask, 
		   counts_ip:np.ndarray, 
		   sum_counts_ip:np.ndarray, 
		   counts_in:np.ndarray = None, 
		   sum_counts_in:np.ndarray = None):
	""" """
	if mask is not None and type_data == 'cut':
		prop = np.sum(mask) / len(counts_ip)
		sum_count_rep = np.nansum(counts_ip[mask]) / sum_counts_ip
		return (sum_count_rep - prop) / prop * 100, prop

	elif mask is not None and type_data == 'chip':
		ip = np.sum(counts_ip[mask]) / sum_counts_ip
		input = np.sum(counts_in[mask]) / sum_counts_in
		if input == 0:
			return None
		return (ip - input) / input * 100

	else:
		return None


def compute_enr(
		type_data:str, 
		repeats_df:pd.DataFrame, 
		chr_len:dict, 
		counts_ip:np.ndarray, 
		sum_counts_ip:np.ndarray, 
		counts_in:np.ndarray = None, 
		sum_counts_in:np.ndarray = None, 
		frag=None
		):

	rep_pos = repeats_pos(repeats_df, chr_len)
	if type_data == 'chip':
		enr = enrich(type_data, rep_pos, counts_ip, sum_counts_ip, counts_in, sum_counts_in)
		return enr
	else:
		n = reads_inrep(counts_ip, frag, mask=rep_pos)
		enr, prop = enrich(type_data, rep_pos, counts_ip, sum_counts_ip)
		return enr, n, prop


if __name__ == "__main__":
	start_time = time.time()
	print(args.type_data)

	df = pd.read_csv('../../data/T2T/rmsk.bed', sep='\t', header=None)
	df.columns = ['chrom','start','end','name','score','strand','thickStart','thickEnd',
			  'reserved','swScore','repClass','repFamily','repDivergence','linkageID']

	# remove Simple_repeat and chromosome Y
	df = df[(df.repClass.isin(['SINE','LINE','Satellite'])) & (df.chrom != 'chrY')]
	#df = df[(df.repClass != 'Simple_repeat') & (df.chrom != 'chrY')]
	#df = df[(df.name == 'ALR_Alpha') & (df.chrom == 'chr1')]

	if args.type_data == 'chip':
		#rep_reads_ip = np.mean(sf.get_frag(args.bam[0], include_n_chroms=2, ignore_duplicates=True, mapping_qual=[0,1])) / 128 
		#rep_reads_input = np.mean(sf.get_frag(args.bam[1], include_n_chroms=2, ignore_duplicates=True, mapping_qual=[0,1])) / 128 
		rep_reads_ip, rep_reads_input = 250, 250
		#print('frag len')
		chr_len, all_ip, all_input = prepare_data(args.type_data, args.filename)
		sum_counts_ip = np.nansum(all_ip)
		sum_counts_input = np.nansum(all_input)
		print(sum_counts_ip, rep_reads_ip)
		N = np.round(sum_counts_ip / rep_reads_ip).astype(np.int32)
		M = np.round(sum_counts_input / rep_reads_input)
		print(f'N = {N}, M = {M}')

		with Parallel(n_jobs=8, prefer='threads') as par:
			res = par(
				delayed(reads_inrep)(
					all_ip, 
					rep_reads_ip,
					all_input, 
					rep_reads_input,
					repeats_df = df[df.name == repeat],
				) for repeat in df.name.unique()
			)

		res_df = pd.DataFrame(res, columns=['n', 'm'])
		res_df['N'] = N
		res_df['M'] = M
		res_df['name'] = df.name.unique()

		with Parallel(n_jobs=8, prefer='threads') as par:
			res = par(
				delayed(compute_enr)(
					args.type_data,
					df[df.name == repeat],
					chr_len,
					all_ip,
					sum_counts_ip,
					all_input,
					sum_counts_input,
				) for repeat in tqdm(df.name.unique())
			)
		res_df['enrichment'] = res
		res_df['proba'] = res_df.m / M
		

	else:
		rep_reads = np.mean(sf.get_frag(args.bam[0], include_n_chroms=2, ignore_duplicates=True, mapping_qual=[0,1]))
		chr_len, all_gen = prepare_data(args.type_data, args.filename)
		sum_counts = np.nansum(all_gen)
		N = np.round(sum_counts / rep_reads).astype(np.int32)

		with Parallel(n_jobs=8, prefer='threads') as par:
			res = par(
				delayed(compute_enr)(
					args.type_data,
					df[df.name == repeat],
					chr_len,
					all_gen,
					sum_counts,
					frag=rep_reads,
				) for repeat in df.name.unique()
			)

		res_df = pd.DataFrame(res, columns=['enrichment', 'n', 'proba'])
		res_df['N'] = N
		res_df['name'] = df.name.unique()
	
	res_df = res_df.assign(
			pval=lambda x: [
				#stats.binomtest(n, N, p=p, alternative='greater').pvalue 
				#for n, p in zip(np.round(x.n).astype(int), x.proba)
				stats.binomtest(n, N, p=p, alternative='two-sided').pvalue 
				for n, p in zip(np.round(x.n).astype(int), x.proba)
			]
		)

	res_df = res_df.merge(df[['name','repFamily','repClass']].drop_duplicates(), how='left', on='name')
	res_df.to_csv(args.output, sep='\t')
	print(f"--- {(time.time() - start_time)/60:.2f} minutes ---")
