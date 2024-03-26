# DeepAF
DeepAF is a Python package that improves the accuracy of high-throughput AlphaFold-Multimer screens using the 3D structural information learned by a DNN to identify real protein complexes. 

The pipeline consists of four major steps:
1. Obtain predicted protein complexes using an optimized version of [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown/tree/main).
2. Calculate 3D representations of each protein complex using Euclidian distance-based encoding.
3. Train the DenseNet3D model (implementation based on https://github.com/GalDude33/DenseNetFCN-3D).
4. Run the query dataset.

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

python3 -m pip install git+git://github.com/anjali-nelliat/AlphaPulldown.git@main
pip install jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

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
The pickle files generated in Step 1 are supplied as monomer objects for structure prediction on GPU. Thee ```--mode``` option indicates the nature of interactions as described in the [AlphaPulldown documentation](https://github.com/KosinskiLab/AlphaPulldown/tree/main)
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
--dataset [PATH to the CSV dataset] \
--data_dir [PATH to AlphaPulldown prediction results folder] \
--output_dir [PATH to folder to save tensors and data files] \
--relaxed \  #Use relaxed AlphaPulldown predictions
--threads [Number of threads to run] \
```
Refer to the ```create_tensors.py``` script for additional modifiable parameters.
The output includes 3D tensors of complexes as individual ```.npy``` files and a csv file which lists the complexes and their class label (i.e 1 for 'positive' and 0 for NA or 'negative'). 

### Step 5: Train model
Train the model on 3D tensors of known complexes and test model on validation dataset. 
```
python train_model.py \
--datapath [PATH to data tensors] \
--weights [PATH to weights in case weights already exist from previous training and need to be fine-tuned] \
--savingPath [PATH to save trained models] \
--train_set [PATH to train set csv file generated by ] \
--test_set [PATH to validation set csv file generated in preprocess]
```
Refer to the ```train_model.py``` script for additional modifiable parameters.

The output is an ```best_model.hdf5``` file with updated weights.  

### Step 6: Query model
Run trained model on query data tensor.
```
python query_model.py \
--datapath [PATH to data tensors] \
--weights [PATH to weights] \
--output [PATH to save prediction results] \
--query_data [PATH to test set csv file generated in preprocess]
```
Refer to the ```query_model.py``` script for additional modifiable parameters.


