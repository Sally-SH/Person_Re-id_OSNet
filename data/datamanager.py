from __future__ import division, print_function, absolute_import
import torch

from data.sampler import build_train_sampler
from data.market1501 import Market1501
from data.transforms import build_transforms


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError('sources must not be None')

        if self.targets is None:
            self.targets = self.sources

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        load_train_targets (bool, optional): construct train-loader for target datasets.
            Default is False. This is useful for domain adaptation research.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        train_sampler_t (str, optional): sampler for target train loader. Default is RandomSampler.
    """
    data_type = 'image'

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler='RandomSampler',
    ):

        super(ImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        print('=> Loading train (source) dataset')
        trainset = Market1501(
            transform=self.transform_tr,
            k_tfm=k_tfm,
            mode='train',
            root=root,
            split_id=split_id
        )

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        name = self.targets
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
        }

        # build query loader
        queryset = Market1501(
            transform=self.transform_te,
            mode='query',
            root=root,
            split_id=split_id
        )
        self.test_loader[name]['query'] = torch.utils.data.DataLoader(
            queryset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        # build gallery loader
        galleryset = Market1501(
            transform=self.transform_te,
            mode='gallery',
            verbose=False,
            root=root,
            split_id=split_id
        )
        self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
            galleryset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        self.test_dataset[name]['query'] = queryset.query
        self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source images   : {}'.format(len(trainset)))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')

