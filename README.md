# DeepAF
DeepAF is a Python package that predicts protein-protein interactions and identifies functional clusters. DeepAF improves the accuracy of high-throughput AlphaFold-Multimer screens using the 3D structural information learned by a Dense Convolutional Neural Network (DenseNet) to identify real protein complexes. 
The high-confidence interactors are then clustered by potential functional groups/pathways using Knowledge Graph embeddings calculated from running the edge2vec algorithm on interactions from the Depmap, Bioplex, OpenCell and BioGRID databases.

![alt text](https://github.com/anjali-nelliat/DeepAF/blob/main/assets/DeepAF_workflow.png)

## Pre-installation
Download Alphafold databases as decribed in the [AlphaFold documentation](https://github.com/google-deepmind/alphafold)

## Installation
### Create Anaconda environment
Create an environment with required dependencies
```
conda create -n DeepAF -c omnia -c bioconda -c conda-forge python==3.10 openmm==8.0 pdbfixer==1.9 kalign2 cctbx-base pytest importlib_metadata hhsuite
```
Activate the conda environment and install HMMER and the modified version of AlphaPulldown with parallel MSA execution enabled.
```
source activate DeepAF
conda install -c bioconda hmmer

python3 -m pip install -e git+https://github.com/anjali-nelliat/AlphaPulldown.git@main
pip install jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python3 -m pip install -e git+https://github.com/anjali-nelliat/DeepAF.git@main

```

## Running DeepAF
### Step 1: Generate MSAs
Use AlphaPulldown to construct MSAs for batch multimer prediction. The output is msa pickle files with the same names as the descriptions of the sequences in fasta files.
```
source activate DeepAF
create_individual_features.py \
  --fasta_paths=baits.fasta,example_1_sequences.fasta \
  --data_dir=<path to alphafold databases> \
  --save_msa_files=False \
  --output_dir=<dir to save the output objects> \ 
  --use_precomputed_msas=False \
  --max_template_date=<any date you want, format like: 2050-01-01> \
  --skip_existing=False \
  --seq_index=<any number you want or skip the flag to run all one after another>
```

### Step 2: Predict complex structures
The pickle files generated in Step 1 are supplied as monomer objects for structure prediction on GPU. The ```--mode``` option indicates the nature of interactions as described in the [AlphaPulldown documentation](https://github.com/KosinskiLab/AlphaPulldown/tree/main)
```
run_multimer_jobs.py --mode=pulldown \
--num_cycle=3 \
--num_predictions_per_model=1 \
--output_path=<output directory> \ 
--data_dir=<path to alphafold databases> \ 
--protein_lists=baits.txt,candidates.txt \
--monomer_objects_dir=/path/to/monomer_objects_directory \
--job_index=<any number you want>
```

### Step 3: Generate a CSV file with AlphaFold confidence metrics
The singularity image can be downloaded from [alpha-analysis.sif](https://drive.google.com/file/d/1FzvFf9FaMG0_kZSHAHdnkxWq-K4pz4F7/view?usp=sharing). 
```
singularity exec \
    --no-home \
    --bind /path/to/your/output/dir:/mnt \
    <path to your downloaded image>/alpha-analysis.sif \
    run_get_good_pae.sh \
    --output_dir=/mnt \
    --cutoff=10
```
cutoff is to check the value of PAE between chains. In the case of multimers, only models with inter-chain PAE values smaller than the cutoff are reported.

### Step 4: Create 3D tensor representations of complexes identified by AlphaPulldown
The input dataset should be a csv file in the following format:
```
name,type
P02994_and_P53303,NA
P02994_and_P53303,NA
.
.
```
where name is the name of the complex folder generated by AlphaPulldown and type can be either NA, or in the case of training data, 'positive' or 'negative'.

```
python create_tensors.py \
--dataset <path to the CSV dataset> \
--data_dir <path to AlphaPulldown prediction results folder> \
--output_dir <path to folder to save tensors and data files> \
--relaxed \  #Use relaxed AlphaPulldown predictions
--threads <Number of threads to run> \
```
Refer to the ```create_tensors.py``` script for additional modifiable parameters.
The output includes 3D tensors of complexes as individual ```.npy``` files and a csv file which lists the complexes and their class label (i.e 1 for 'positive' and 0 for NA or 'negative'). 

### Step 5: Train model
Train the model on 3D tensors of known complexes and test model on validation dataset. 
```
python train_model.py \
--datapath <path to data tensors folder> \
--weights <path to weights in case weights already exist from previous training and need to be fine-tuned> \
--savingPath <path to save trained models> \
--train_set <path to train set csv file generated in the tensor building step> \
--test_set <path to validation set csv file generated in the tensor building step>
```
Refer to the ```train_model.py``` script for additional modifiable parameters.

The output is an ```best_model.hdf5``` file with updated weights.  

### Step 6: Query model
Run trained model on query data tensor.
```
python query_model.py \
--datapath <path to data tensors folder> \
--weights <path to the model> \
--output <file path to save prediction results> \
--query_data <path to query csv file generated in the tensor building step>
```
The output is a numpy array file with (probability of non-interaction, probability of interaction) pairs for each complex. Refer to the ```query_model.py``` script for additional modifiable parameters.

### Step 7: Calculate knowledge graph embeddings using edge2vec
The input file is an interaction matrix with five columns Node 1, Node 2, Edge Type, Edge Weight, Edge ID. All column values are encoded as integers (except Edge Weight which is a floating point number).
In our study, interacting gene/protein pairs from BioGRID, Bioplex and OpenCell, as well as genetic codependencies from Depmap were used to construct the input data.
```
python transition.py \
--input <path to input txt file> \
--output <path to output transition matrix> \
--type_size <number of edge types> \
--em_iteration <number of EM iterations for the transition matrix> \
--e_step <E step in the EM algorithm: there are four metrics to calculate edge type similarity - 1:Wilcoxon 2:Entropy 3:Spearman correlation (default) 4:Pearson correlation \
--walk-length <length of walk per source> \
--num-walks <number of walks per source> \
--weighted # add argument for weighted graph
```
The output is a .txt file with the transition matrix. Refer to the ```transition.py``` script for additional modifiable parameters. The subsequent code is to obtain the final embeddings from the transition matrix

```
python edge2vec.py \
--input <path to input txt file> \
--output <path to output transition matrix> \
--type_size <number of edge types> \
--em_iteration <number of EM iterations for the transition matrix> \
--e_step <E step in the EM algorithm: there are four metrics to calculate edge type similarity - 1:Wilcoxon 2:Entropy 3:Spearman correlation (default) 4:Pearson correlation \
--dimensions <number of dimensions> \
--walk-length <length of walk per source> \
--num-walks <number of walks per source> \
--p <return hyperparameter> \
--q <in-out hyperparameter> \
```
The output is a .txt file with the final embeddings for the knowledge graph. Refer to the ```edge2vec.py``` script for additional modifiable parameters.
These embeddings are projected into 2D space using a PCA and the high-confidence interactors of our protein of interest (iPTM >0.6, probability of interaction > 0.6) are highlighted to identify clusters of interactors.

This method was successful in identifying multiple novel protein chaperones and a highly conserved chaperone system when interfaces with data from [HSF1base](https://hsf1base.org/)


