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


