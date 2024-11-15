python train.py -m training.project=CrowdMAC_SDD      dataset.datasets=[stanford]       model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_IND-TIME dataset.datasets=[ind-time-split] model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_FDST     dataset.datasets=[fdst]           model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_VSCROWD  dataset.datasets=[vscrowd]        model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_JRDB     dataset.datasets=[jrdb]           model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_HT21     dataset.datasets=[ht21]           model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none
python train.py -m training.project=CrowdMAC_ETHUCY   dataset.datasets=[eth],[univ],[hotel],[zara1],[zara2] model.checkpoint=pretrained/sdd-pretrained.pth training.epochs=200 training.test_per=10 optimizer.lr=0.00005 optimizer.min_lr=0.000005 optimizer.warmup_epochs=10 training.pred_mask.mask_type=forecastfuture training.obs_mask.mask_type=none