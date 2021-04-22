## CS1430 Final

![alt text](https://github.com/artset/cs1430-final/blob/main/writeup/results.png?raw=true)


This project is a reimplementation of the the [following paper](https://arxiv.org/pdf/2002.05638.pdf) on Image to Illustration translation using GANs for Brown's CS1430
Computer Vision course, created in Spring 2021. This model is used to translate
landscape photographs into illustrations inspired by Studio Ghibli films.

The private repository for this model can be found [here](https://github.com/artset/cs1430-final-data). Please reach out to gain access!


#### How to run
The model can be found in `model/ganilla.py`. To run the model, one must make a `data` directory in the root directory, with train and test directories. To load the model with
checkpoints, run `python run.py --load-checkpoints {directory to checkpoints}`. The model
will automatically save checkpoints every 10 epochs in `checkpoints/{data_name}/{time_stemp}/epoch_{epoch_number}`.


#### How to evaluate

1) If you're running locally, download the corresponding g1, g2, d1, d2 weights from the VM.Weights should be put in the directory: `checkpoints/{dataset}/epoch_{epoch}`. Otherwise skip this step


2) In run.py, edit the following constants at the top:

```
DATASET_NAME = {dataset}

EPOCH = {epoch}

SAVE_COUNT = 5
```

  Update DATASET name as this determines the directory of the saved images.
  Update epoch count to the valid epoch so it saves in the proper folder.
  You can keep SAVE_COUNT as 5 to generate 5 generated images and 5 cycled images.

3) Type the following in the command line w/ the valid filepath to the checkpoint folder.
Note this folder should contain 4 sets of h5 files (g1, g2, d1, d2)

If you reformatted it diff and are doing locally:
`python run.py --load-checkpoint checkpoints/{dataset}/epoch_{epoch} --evaluate`


You might have to do this if you're on the VM:
`python run.py --load-checkpoint checkpoints/{timestamp}/epoch_{epoch} --evaluate`


4) It should generate images in the correct directories as well as print the psnr, ssim, and mse.

#### Contributions
- [x] GCP set up VM + Buckets [Katherine]
- [x] Preprocessing 
  - [x] Write script to scrape from OpenLibrary [Minna, Katherine, Liyaan, Zoe]
  - [x] Scrape illustration data from OpenLibrary [Katherine]
  - [x] Get landscape data from CycleGAN dataset/Kaggle [Zoe, Liyaan, Minna]
  - [x] Scrape Miyazaki images from Google Images [Minna, Katherine]
  - [x] Write preprocessing script to prepare images for model  [Minna, Katherine]
- [x] Model 
  - [x] Write script to run pipeline [Minna, Katherine]
  - [x] Set up model checkpoints and loading weights [Minna, Katherine]
  - [x] Model architecture
    - [x] Generator [Zoe, Liyaan]
    - [x] Discriminator [Minna, Katherine]
    - [x] Ganilla model that puts everything together [Katherine]
- [x] Progress Report [Minna, Katherine, Liyaan]
- [x] Training Model
  - [x] Miyazaki [Minna, Katherine]
  - [x] Elmer  - simple generator and complex generator [Liyaan, Zoe]
- [x] Evaluation script
  - [x] Reconstruction Metrics [Minna, Katherine]
  - [x] Generate Sample images [Minna, Katherine]
- [x] Presentation slides [All]
- [x] Final Write Up [Minna, Katherine]
