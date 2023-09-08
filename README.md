## Requirements
python3  
pytorch  
scipy  
[chumpy](https://github.com/mattloper/chumpy)  
[psbody.mesh](https://github.com/MPI-IS/mesh)

Code works with psbody.mesh v0.4 , pytorch >= v1.0 , chumpy v0.7 and scipy v1.3 .

## How to Run
- Download and prepare SMPL model and TailorNet data from [dataset repository](https://github.com/zycliao/TailorNet_dataset).
- Set DATA_DIR and SMPL paths in `global_var.py` file accordingly.
- Download trained model weights in a directory and set its path to MODEL_WEIGHTS_PATH variable in `global_var.py`.
  - [old-t-shirt_female_weights](https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female_weights.zip)
        (4.1 GB)
  - [t-shirt_male_weights](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male_weights.zip)
        (2.0 GB)
  - [t-shirt_female_weights](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female_weights.zip)
        (2.0 GB)
  - [shirt_female_weights](https://datasets.d2.mpi-inf.mpg.de/tailornet/shirt_female_weights.zip)
        (2.5 GB)
  - [shirt_male_weights](https://datasets.d2.mpi-inf.mpg.de/tailornet/shirt_male_weights.zip)
        (2.5 GB)
  - [This](https://nextcloud.mpi-klsb.mpg.de/index.php/s/LTWJPcRt7gsgoss) link has all the weights listed above as well as the following:
    - pant_female_weights
    - pant_male_weights
    - short-pant_female_weights
    - short-pant_male_weights
    - skirt_female_weights
- Set output path in `run_tailornet.py` and run it to predict garments on some random inputs. You can play with 
  different inputs. You can also run inference on motion sequence data.
- To visualize predicted garment using blender, run `python run_tailornet.py render`. (Blender 2.79 needs to be installed.)

## TailorNet Per-vertex Error in mm on Test Set
... evaluated using `evaluate` function in `utils/eval.py`.
| garment_class | gender | TailorNet Baseline | TailorNet Mixture Model |
|:--:|:--:|:--:|:--:|
|  old-t-shirt  | female | 11.1 | 10.7 |
|      t-shirt  | female | 12.6 | 12.3 |
|      t-shirt  |   male | 11.4 | 11.2 |
|        shirt  | female | 14.2 | 14.1 |
|        shirt  |   male | 12.7 | 12.5 |
|         pant  | female |  4.7 |  4.8 |
|         pant  |   male |  8.1 |  8.1 |
|   short-pant  | female |  6.8 |  6.6 |
|   short-pant  |   male |  7.0 |  7.0 |
|        skirt  | female |  7.7 |  7.8 |

## Training TailorNet yourself
- Set global variables in `global_var.py`, especially LOG_DIR where training logs will be stored.
- Set config variables like gender and garment class in `trainer/base_trainer.py` (or pass them via command line)
and run `python trainer/base_trainer.py` to train TailorNet MLP baseline.
- Similarly, run `python trainer/lf_trainer.py` to train low frequency predictor and `trainer/ss2g_trainer.py` to
train shape-style-to-garment(in canonical pose) model.
- Run `python trainer/hf_trainer.py --shape_style <shape1>_<style1> <shape2>_<style2> ...` to train pivot high 
frequency predictors for pivots `<shape1>_<style1>`, `<shape2>_<style2>`, and so on. See 
`DATA_DIR/<garment_class>_<gender>/pivots.txt` to know available pivots.
- Use `models.tailornet_model.TailorNetModel` with appropriate logdir arguments to do prediction.


