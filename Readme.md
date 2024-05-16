# NIRec
## Environment
We provide the environment that our codes depend on. To install the conda environment, run:
```
conda env create -f ./pyg_usual.yaml
```
## Simulation
We provide our simulation regarding Ciao. The synthetic datasets and the constructed target users (items) are located in ./datasets/ directory. The simulated user and item representations are located in ./simulation/para/ directory.


## Run the Code
We provide our codes for adjusting the exposure of the target item to the target users. We can assign different model names for the parameter click_model_name to get adjustments based on different models. Run the following command line, we get IoIP and DtNE given a group size of 50, a neighbor interference strength coefficient of 10, and an individual interest threshold of 1.
```
python ./code/group_main.py --data_name=ciao --core=10 --alpha1=0.1 --alpha2=0.2 --alpha3=0.7 --beta=10.0 --power=0.5 --sample_type=ph --is_rm=0 --prefer_thres=1.0 --num_neigh_thres=0 --num_guided_item=1000 --num_group=1 --group_size=50 --seed=0 --click_model_name=gatv2_spill_var2
```
Run ./code/merge_results_save_revision.ipynb to get data for IoIP-DtNE curve.
Run ./code/plot_revision.ipynb to plot IoIP-DtNE curve.