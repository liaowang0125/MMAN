import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import cv2
from PIL import Image
# from util.visualizer import Visualizer
# from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)
## create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    for k,v in visuals.items():
    	name=os.path.join('output','_'.join([str(i),k,'.jpg']))
    	# image_pil = Image.fromarray(v)
    	# print(k,image_pil.size)
    	v.save(name)
    	# cv2.imwrite(name,v)
    # img_path = model.get_image_paths()
    print('process image... %d' % i)
    # visualizer.save_images(webpage, visuals, img_path)

# webpage.save()
