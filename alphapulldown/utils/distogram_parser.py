__author__ = "Grzegorz Chojnowski"
__email__ = "gchojnowski@embl-hamburg.de"
__date__ = "16 Apr 2024"
__info__ = "extracts contact information from AF2 distograms"


import pickle, glob
import os
import numpy as np
import string

class distogram_parser:
    """
        extracts predicted contacts from AF2 distograms assuming that 
        'seq' list in output pickle maps onto the distogram 
            (contains all target sequences in output model/distogram order)

        output: list of residues close in space [(res_A_idx, res_A_chain), (res_B_idx, res_B_chain), pbty( dist(A,B)<distance ) > pbtycutoff ]
    """

    def __init__(self):
        pass

    def get_contacts(self, directory, distance=8, pbtycutoff=0.8, cross_only=True, verbose=False):
        """
            selects from datadir a pkl/distogram corresponding to a top-ranked model 
        """

        top_ranked_dgram = (None, None, 0.0)
        for fn in glob.glob(os.path.join(datadir, "*.pkl")):
            with open(fn, 'rb') as ifile:
                d=pickle.load(ifile)
            if d.get('ranking_confidence',0)>top_ranked_dgram[-1]:
                top_ranked_dgram = (fn, d, d.get('ranking_confidence',0))

        if top_ranked_dgram[0] is None: return []

        if verbose:
            print(f"Selected {os.path.basename(top_ranked_dgram[0])} with ranking confidence {top_ranked_drgam[-1]:.2f}")

        d = top_ranked_dgram[1]

        # reparse top ditogram; avoids storing all pickles in memory         
        with open(fn, 'rb') as ifile:
            d=pickle.load(ifile)   
        chain_ids = string.ascii_uppercase
        asym_id=[]
        chain_lens = []
        for _idx,_seq in enumerate(d['seqs']):
            asym_id.extend([_idx+1]*len(_seq))
            chain_lens.append(len(_seq))
        chain_lens = np.array(chain_lens)

        assembly_num_chains = len(d['seqs'])

        bin_edges = d['distogram']['bin_edges']

        # apply softmax (scipy equivalent)
        x = d['distogram']['logits']
        _x = np.exp(x - np.amax(x, axis=-1, keepdims=True))
        probs =  _x / np.sum(_x, axis=-1, keepdims=True)

        #_probs = scipy.special.softmax(d['distogram']['logits'], axis=-1)
        #assert (probs == _probs).all()


        distance = np.clip(distance, 3, 20)

        bin_idx=np.max(np.where(bin_edges<distance))

        below_dist_pbty = np.sum(probs, axis=2, where=(np.arange(probs.shape[-1])<bin_idx))

        requested_contacts=[]

        resi_i,resi_j = np.where(below_dist_pbty>pbtycutoff)
        for i,j in zip(resi_i, resi_j):

            ci = int(asym_id[i]-1)
            cj = int(asym_id[j]-1)

            # skipp: self, intra-chain (default), and symm
            if i==j: continue
            if cross_only and ci==cj: continue
            if ci>cj: continue

            reli = 1+i-sum(chain_lens[:ci])
            relj = 1+j-sum(chain_lens[:cj])


            if verbose: print(f"{reli:-4d}/{chain_ids[ci]} {relj:-4d}/{chain_ids[cj]} {below_dist_pbty[i,j]:5.2f}")

            requested_contacts.append([(reli,chain_ids[ci]), (relj,chain_ids[cj]),  below_dist_pbty[i,j]])

        return(requested_contacts)


if __name__=="__main__":

    do=distogram_parser()
    contacts=do.get_contacts(datadir='.', verbose=0)


