# pretraining for medical image segmentation

Following SwinUNETR and SwinUnet...

For image inpainting as self-supervised pre-training, randomly cropped tokens are augmented. Model is to fill the randomly generated mask for each images.
![image](https://github.com/hyukiggle/pretraining/assets/49806099/39af932d-f781-4369-9fe0-781b5054a470)
![image](https://github.com/hyukiggle/pretraining/assets/49806099/c2410494-eb74-498a-915b-c6376e168dd1)

Since this project is built for pre-training, this repository only include basice pipeline of image inpainting, not techinques for better results.
Dataset for this project is ImageNet.
