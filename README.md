## CS1430 Final

This project is a reimplementation of the the [following paper](https://arxiv.org/pdf/2002.05638.pdf) on Image to Illustration translation using GANs for Brown's CS1430
Computer Vision course, created in Spring 2021. 


The data for this model can be found [here](https://github.com/artset/cs1430-final-data)


#### TODO
- [x] GCP set up VM + Buckets [Katherine]
- [x] Preprocessing 
  - [x] Write script to scrape from OpenLibrary [Minna, Katherine, Zoe, Liyaan]
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
- [ ] Training Model
  - [ ] Miyazaki [Minna, Katherine]
  - [ ] Elmer
- [ ] Evaluation script
  - [ ] Reconstruction Metrics
  - [ ] Generate Sample images
- [ ] Presentation slides
- [ ] Record Demo
- [ ] Final Write Up




#### To run evaluation:

python run.py --load-checkpoint checkpoints/{dataset}/epoch_{epoch} --evaluate

Make sure to comment in 
reconstruction_eval or generate_illo
depending on the evaluation task you would like to perform.

dataset options: elmer, miyazaki