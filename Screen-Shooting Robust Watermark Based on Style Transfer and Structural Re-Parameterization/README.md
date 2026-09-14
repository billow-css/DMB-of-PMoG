# To train the network 
python main.py --dataset train_mask --mode train_mask --image_dir '.../' --image_val_dir '.../'

PS: Before training, pls generate the training dataset like the examples shown in the 'Datasets/COCOMask/train/train_class/'

# To use the pre-trained model to embed watermark
python main.py --dataset test_embedding --mode test_embedding --image_dir '.../'  --embedding_epoch 99 --distortion ScreenShooting

# To test the accuracy of the network
python main.py --dataset test_accuracy --mode test_accuracy --image_dir '.../' --image_val_dir '.../' --embedding_epoch 99 --distortion ScreenShooting

PS: After screen shooting process, please first utilize the ``PespectiveTransformation.m'' to make a perspective correction. Besides, when capturing the watermarked image, enlarge it and make it occupying at least 1 / 4 of the screen for better performance.

# Example:

嵌入：python main.py --dataset test_embedding --mode test_embedding --image_dir Datasets/test_images/  --embedding_epoch 96 --distortion ScreenShooting

python main.py --dataset test_accuracy --mode test_accuracy --image_val_dir Datasets/cut_images/  --embedding_epoch 96 --distortion ScreenShooting

嵌入分割图像：python main.py --dataset test_embedding --mode test_embedding --image_dir Datasets/cut1/  --embedding_epoch 96 --distortion ScreenShooting
解码：python main.py --dataset test_accuracy --mode test_accuracy --image_dir Datasets/qianru/ --embedding_epoch 96 --distortion ScreenShooting