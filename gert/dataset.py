from torch.utils.data import Dataset
import tqdm
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import glob
import collections
from util import draw_exclusive, get_context, get_seq, GeneInterval, mask, mutate


def get_operon(genes, idx, window_size):
    """ Retrieves genes within the window size surrounding gene

    Parameters
    ----------
    genes : list of GeneInterval
        List of genes
    idx : int
        Index of gene of interest
    window_size : int
        Size of window surrounding gene.
    """
    s, e = genes[idx].start, genes[idx].end
    coord = (s + e) / 2
    lidx = max(0, idx - 1)
    ridx = min(idx + 1, len(genes))

    while (coord - genes[lidx].end) < window_size and lidx > 0:
        lidx = lidx - 1

    while (genes[ridx].start - coord) < window_size and ridx < len(genes):
        ridx = ridx + 1

    return genes[lidx : idx] + genes[idx + 1 : ridx]


class ExtractIntervals(object):
    def __init__(self, window_size=10000):
        self.window_size = window_size

    def __call__(self, gb_file):
        gb_record = SeqIO.read(open(gb_file, "r"), "genbank")
        cds = list(filter(lambda x: x.type == 'CDS', gb_record.features))
        starts = list(map(lambda x: int(x.location.start), cds))
        ends = list(map(lambda x: int(x.location.end), cds))
        strand = list(map(lambda x: x.strand, cds))
        seqs = list(map(get_seq, cds))
        res = zip(starts, ends, seqs, strand)

        # sequences with start, end and position
        res = list(filter(lambda x: len(x) > 0, res))
        res = list(map(lambda x: GeneInterval(
            start=x[0], end=x[1], sequence=x[2], strand=x[3]
        ), res))
        return {'gene_intervals': res}


class SampleGenes(object):
    def __init__(self, num_sampled, within_prob=0.5, window_size=10000):
        """ Randomly samples genes within genome

        Parameters
        ----------
        num_sampled : int
            Number of genes sampled per genome
        within_prob : float
            The probability of drawing a gene within the same operon
        window_size : int
            The radius of the operon (typically around 10kb)
        """
        self.num_sampled = num_sampled
        self.within_prob = within_prob
        self.window_size = window_size

    def __call__(self, record):
        """ Randomly samples genes are their paired genes

        Parameters
        ----------
        record: dict
           key : 'gene_intervals'
           values : list of GeneInterval objects
        """
        gis = record['gene_intervals']
        # draw genes
        idx = list(np.random.randint(0, len(gis), size=self.num_sampled))
        draws = list(np.random.random(size=self.num_sampled))
        def draw_gene(i, d):
            if d <= self.within_prob:
                # draw operon
                operon = get_context(gis, i, self.window_size)
                j = np.random.randint(0, len(operon))
                return gis[i], operon[j]
            else:
                # draw outside operon
                j = draw_exclusive(len(gis), i)
                return gis[i], gis[j]
        pairs = list(map(lambda x: draw_gene(x[0], x[1]), zip(idx, draws)))
        genes, next_genes = zip(*pairs)
        return {'genes' : genes, 'next_genes' : next_genes}


class MaskPeptides(object):
    def __init__(self, mask_prob=0.8, mutate_prob=0.25,
                 mask_chr='_'):
        """ Masks peptides

        Parameters
        ----------
        mask_prob : float
           Probability of masking a peptide
        mutate_prob : float
           Probability of mutation

        Notes
        -----
        Is it worthwhile to think about throwing a blosum matrix?
        """
        self.mask_prob = mask_prob
        self.mutate_prob = mutate_prob
        self.mask_chr = mask_chr
        self.vocab = PeptideVocab()

    def __call__(self, sample):
        gene = np.array(list(sample['gene'].sequence))
        next_gene = np.array(list(sample['next_gene'].sequence))
        gene_seq = mutate(gene, self.mutate_prob, self.vocab)
        gene_seq = mask(gene_seq, prob=self.mask_prob, mask_chr=self.mask_chr)
        return {'gene_seq': gene_seq, 'next_seq': next_gene}
