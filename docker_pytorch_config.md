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



