## Lessons Learned

- A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted.

- The standard color augmentation in [21] is used.

- batch normalization (BN) right after each convolution and before activation, following [16]

- "We use SGD with a mini-batch size of 256."
	- Accumulate gradients in optimizer to achieve an equivalent batch size of 64
	- My Small Orin Nano would crash with a memory of 8GB. Which should be large enough?? TODO

- During training, accuracy was 70%, but in eval mode, the same dataset gives 17%??

- TODO:
    1. download some more pictures
    2. Transfer Learning with Resnet 101, and CIFAR-100?

## Questions

- Data transformation: apply the same trasformation to test and train dataset?? TODO

- Simuilate a batch: do I need to divide loss by batch size?
            # This is because torch.nn.CrossEntropyLoss(reduction='mean') is true, so to simulate a larger batch, we need to further divide
            loss = criterion(outputs, labels)/ACCUMULATION_STEPS


- Why convert pixel values to [0,1]?

- What does `train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)` do? Pinned-memory for faster transfer?

- How to monitor jetson GPU usage? Jtop
- Mixed precision training?? TODO

- Learning Rate = 0.01 is quite high for Adam, 0.1 is good for `SGD`

- `class_names = train_data.classes`

## UNet Questions:

### UNet Low Training Accuracy

- PyTorch Net:  ERROR Failed to serialize metric: division by zero??


- tqdm?` from tqdm import tqdm` I love the summaries

- Tried SGD
- Tried data with `nn.CrossEntropyLoss(ignore_index)`. The final learning is bad. 
    - Do I need to subtract the avg and std dev in test set? 
- Try adding batch norm in conv layers.

- RuntimeError: Mismatched Tensor types in NNPack convolutionOutput: This is because there's a mis match in types
    - One hidden bug could be not having `inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the correct device`, so inputs and labels are not on the same device
- "skip downloading"
    ```python
    def is_extracted(dataset_dir, year='2007'):
        extracted_train_path = os.path.join(dataset_dir, 'VOCdevkit', f'VOC{year}', 'ImageSets', 'Segmentation', 'train.txt')
        return os.path.exists(extracted_train_path)

    self._dataset = datasets.VOCSegmentation(
        root=DATA_DIR,  # Specify where to store the data
        year='2007',    # Specify the year of the dataset (2007 in this case)
        image_set=image_set,  # You can use 'train', 'val', or 'trainval'
        download=not is_extracted(dataset_dir=DATA_DIR),  # Automatically download if not available
        transform=image_seg_transforms,  # Apply transformations to the images
        target_transform=target_seg_transforms  # Apply transformations to the masks
    )
    ```
- Training Error:
    ```python
      File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/loss.py", line 1179, in forward
        return F.cross_entropy(input, target, weight=self.weight,
      File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 3059, in cross_entropy
        return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
    RuntimeError: CUDA error: device-side assert triggered
    CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
    Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
    ```
    - If you truly want to debug this, you need to recompile torch with `TORCH_USE_CUDA_DSA`. 
    - `CUDA_LAUNCH_BLOCKING` only throws the error synchronously from CUDA. 
    - **So it's a pain to debug in this case**
    - One cause of this error could be mismatching of preds and y. In my case, the offending line is: `loss = nn.CrossEntropy(preds, y)`. 
        - If the output channel doesn't match with the output.

## Things To Try:
- Sigmoid at the end, even tho CrossEntropyLoss handles logits already


## Useful Commands
- torch.unique


## Habits
- WHen done, put the model as part of the release
- constants on top of the file so it's easy to test with different values 

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) This allows us to do FP16 Arithmatic on recent GPUs. Enabling AMP is good.

- Real Time Visualization using [wandb](https://wandb.ai/site)
- Save checkpoints
    ```python
    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict['mask_values'] = dataset.mask_values
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f'Checkpoint {epoch} saved!')
    ```
- Clear torch cache:
    ```
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    ```


