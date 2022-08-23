from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
import numpy as np
import pickle
import torchvision
import torch
import os
import time
import logging
from torchvision import datasets
import matplotlib.image as mpimg
import cv2 #For binanry mask edge detection
import argparse
from tqdm import tqdm

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset,IMG_EXTENSIONS
import tensorflow.compat.v1 as tf

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    tf.compat.v1.disable_eager_execution()
    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class SSLTFDataset(VisionDataset):
    def __init__(self, root: str, extensions = IMG_EXTENSIONS, transform = None):
        self.root = root
        self.transform = transform
        self.samples = make_dataset(self.root, extensions = extensions) #Pytorch 1.9+
        self.coder = ImageCoder()

    def _is_cmyk(self,filename):
        """Determine if file contains a CMYK JPEG format image.

        Args:
        filename: string, path of the image file.

        Returns:
        boolean indicating if the image is a JPEG encoded with CMYK color space.
        """
        # File list from:
        # https://github.com/cytsai/ilsvrc-cmyk-image-list
        cmyk_excluded = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                       'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                       'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                       'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                       'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                       'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                       'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                       'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                       'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                       'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                       'n07583066_647.JPEG', 'n13037406_4650.JPEG']
        return filename.split('/')[-1] in cmyk_excluded

    def _is_png(self,filename):
        """Determine if a file contains a PNG format image.

        Args:
        filename: string, path of the image file.

        Returns:
        boolean indicating if the image is a PNG.
        """
        # File list from:
        # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
        return 'n02105855_2933.JPEG' in filename

    def _process_image(self, filename):

        """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        fh_scales: Felzenzwalb-Huttenlocher segmentation scales.
        fh_min_sizes: Felzenzwalb-Huttenlocher min segment sizes.
        Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
        """
    
        # Read the image file.
        image_data = tf.gfile.GFile(filename, 'rb').read()

        #ZZimport ipdb;ipdb.set_trace()
        # Clean the dirty data.
        if self._is_png(filename):
            # 1 image is a PNG.
            print('Converting PNG to JPEG for %s' % filename)
            image_data = self.coder.png_to_jpeg(image_data)
        elif self._is_cmyk(filename):
            # 22 JPEG images are in CMYK colorspace.
            print('Converting CMYK to RGB for %s' % filename)
            image_data = self.coder.cmyk_to_rgb(image_data)

        # Decode the RGB JPEG.
        image = self.coder.decode_jpeg(image_data)

        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        
        # Load Image
        image = self._process_image(path)
        label = 0
        
        return (torch.tensor(image).permute(2,0,1),label,path)

    def __len__(self) -> int:
        return len(self.samples)
    
    
class Preload_Masks():
    def __init__(self,dataset_dir,output_dir,ground_mask_dir='',mask_type='fh',experiment_name='',
                 num_threads=os.cpu_count(),scale=1000,min_size=1000,segments=[3,3]):
        
        self.output_dir=output_dir
        self.mask_type=mask_type
        self.scale = scale
        self.min_size = min_size
        self.segments = segments
        self.experiment_name = experiment_name
        self.num_threads = num_threads
        self.ground_mask_dir = ground_mask_dir
        self.image_dataset = SSLTFDataset(dataset_dir)
        self.ds_length = len(self.image_dataset)
        self.save_path = os.path.join(self.output_dir,self.experiment_name)
        
    def create_patch_mask(self,image,segments):
        dims=list(np.floor_divide(image.shape[1:],segments))

        mask=torch.hstack([torch.cat([torch.zeros(dims[0],dims[1])+i+(j*(segments[0])) 
                                      for i in range(segments[0])]) for j in range(segments[1])])

        mods = list(np.mod(image.shape[1:],segments))
        if mods[0]!=0:
            mask = torch.cat([mask,torch.stack([mask[-1,:] for i in range(mods[0])])])
        if mods[1]!=0:
            mask = torch.hstack([mask,torch.stack([mask[:,-1] for i in range(mods[1])]).T])

        return mask.int()

    def create_fh_mask(self,image, scale, min_size):
        mask = felzenszwalb(image.permute(1,2,0), scale=scale, min_size=min_size)
        return torch.tensor(mask).int()
    
    def load_ground_mask(self,img_path):
        #assuming masks have ssame name ass images and asre png files
        mask_path = os.path.join(self.ground_mask_dir,os.path.splitext(img_path.split('/')[-1])[0]+'.png')
        mask = torch.tensor(mpimg.imread(mask_path)[:,:,0])
        return mask
    
    def select_mask(self,obj):
        image,label,img_path = obj
        suffix = '_'+self.mask_type+'.pkl'
        name = os.path.join(self.save_path,os.path.splitext('_'.join(img_path.split('/')[-2:]))[0])
        
        if self.mask_type =='fh':
            mask = self.create_fh_mask(image, scale=self.scale, min_size=self.min_size).to(dtype=torch.int16)
        if self.mask_type =='patch':  
            mask = self.create_patch_mask(image,segments=self.segments).to(dtype=torch.int16)
        if self.mask_type =='ground':
            mask = self.load_ground_mask(img_path).to(dtype=torch.int16)
        
        with open(name+suffix, 'wb') as handle:
            pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [img_path,name+suffix]
    
    def pkl_save(self,file,name):
        with open(name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_dicts(self,img_paths,mask_paths):
        self.pkl_save(mask_paths,os.path.join(self.output_dir,self.experiment_name+'_img_to_'+self.mask_type+'.pkl'))
        return
    
    def forward(self):
        try:
            os.mkdir(os.path.join(self.output_dir,self.experiment_name))
        except:
            if not os.path.exists(self.output_dir):
                os.makedirs(os.path.join(self.output_dir,self.experiment_name))
                
        print('Dataset Length: %d  '%(self.ds_length))
        start = time.time()
        img_paths,mask_paths = zip(*Parallel(n_jobs=self.num_threads,prefer="threads")
                                 (delayed(self.select_mask)(obj) for obj in tqdm(self.image_dataset)))
        end = time.time()

        self.save_dicts(img_paths,mask_paths)

        print('Time Taken: %f  '%((end - start)/60))
        
        return 
    
if __name__=="__main__":
    
    mask_loader = Preload_Masks(dataset_dir = '/home/kkallidromitis/data/imagenet/images/train',
                            output_dir = '/home/kkallidromitis/data/imagenet/masks/',
                            mask_type = 'fh',
                            experiment_name = 'train_tf',
                            )
    mask_loader.forward()
