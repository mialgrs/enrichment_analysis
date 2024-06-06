#!/usr/bin/python

import time, argparse, numpy as np
import pyBigWig as pbw
import pandas as pd
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='+', type=str, 
					help='path to ip then input bigwig file for chip')
parser.add_argument('-b', '--bin', type=int, 
					help='bin size')
parser.add_argument('-o', '--output', type=str, 
					help='filename of the output (Dataframe)')
args = parser.parse_args()


def prepare_data(filename:List[str], bin:int):

		ip_f = pbw.open(filename[0])
		chr_len = ip_f.chroms()
		chr_len.pop('chrY', None)
		chr_len = {name:int(np.ceil(val/bin)) for name, val in chr_len.items()}
		print(chr_len)
		return chr_len


def repeats_pos(repeats_df, chr_len:dict, bin:int):
	pos = []
	val = 0 #add previous chr len to have an unique positions array
	for chr in chr_len.keys():
		dtf = repeats_df[(repeats_df.chrom == chr)]
		if not dtf.empty:
			start = np.ceil(val + (dtf.start)/bin)
			end = val + (dtf.end)//bin
			a = (np.unique(
				np.concatenate(
					[np.arange(i,j + 1) if i < j + 1 else np.empty(shape=0) for i,j in zip(start, end)])
					) - 1).astype(np.int32)
			a = list(a)
			pos = pos + a
		val += chr_len[chr]

	return pos


if __name__ == "__main__":
	start_time = time.time()

	df = pd.read_csv('../../data/T2T/rmsk.bed', sep='\t', header=None)
	df.columns = ['chrom','start','end','name','score','strand','thickStart','thickEnd',
			  'reserved','swScore','repClass','repFamily','repDivergence','linkageID']

	# remove Simple_repeat and chromosome Y
	
	df = df[(df.repClass != 'Simple_repeat') & (df.chrom != 'chrY')]
	df = df[df.repClass.isin(['LINE', 'SINE', 'Satellite'])]
	#df = df[(df.name.isin(['ALR_Alpha', 'AluJb'])) & (df.chrom.isin(['chr1', 'chr2']))]
	chr_len = prepare_data(args.filename, args.bin)
	print(np.sum([chr_len.values]))
	print(len(df))

	nber = 0
	with open(args.output, 'w') as outfile:
		for repeat in df.name.unique():
			mask = repeats_pos(df[df.name==repeat], chr_len, args.bin)
			for val in mask:
				outfile.write(f'{nber}\t{str(val)}\t{np.int8(1)}\n')
			nber += 1

	print(f"--- {(time.time() - start_time)/60:.2f} minutes ---")
