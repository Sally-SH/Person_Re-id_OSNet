# Person_Re-id_OSNet

## Train
Run main.py by this command
```
python main.py
```
For data transformation, change parameter in ImageDataManager
```python
# load dataset
datamanager = ImageDataManager(
    root='dataset',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)
```
If you want decay lerning rate in specific epoch, give the epoch list as the `step size` parameter.
```python
#function for decay learning rate
scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[30,50]
)
```
## Test
Run main.py by this command
```
python main.py
```
Make sure set `test_only=True`

If you set `visrank=True`, The top 10 similar gallery images for a given query will be saved to a file.
```python
engine.run(
    save_dir='log/osnet',
    max_epoch=60,
    eval_freq=10,
    print_freq=20,
    test_only=True,
    visrank=True,
    open_layers='classifier'
)
```
