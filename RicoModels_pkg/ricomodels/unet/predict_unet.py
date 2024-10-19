#! /usr/bin/env python3
import torch
from ricomodels.utils.data_loading import VOCSegmentationClass, GTA5Dataset
from ricomodels.utils.visualization import visualize_image_target_mask
from ricomodels.unet.train_unet import MODEL_PATH, INTERMEDIATE_BEFORE_MAX_POOL
from ricomodels.unet.unet import UNet
from ricomodels.utils.losses import dice_loss, DiceLoss, focal_loss
import time

BATCH_SIZE = 16

@torch.inference_mode()
def eval_model(model, test_dataloader, device, class_num, visualize: bool = False):
    # Evaluation phase
    num_images = len(test_dataloader)
    model.eval()
    correct_test = 0
    total_test = 0

    i = 0
    with torch.no_grad():
        # TODO I AM ITERATING OVER TRAIN_LOADER, SO I'M MORE SURE
        for inputs_test, labels_test in test_dataloader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            _, predicted_test = outputs_test.max(1)
            mask = (labels_test == labels_test)
            local_total = mask.sum().item()
            local_correct = (predicted_test.eq(labels_test) & mask).sum().item()
            total_test += local_total
            correct_test += local_correct

            # labels_test: (m, h, w) 
            if visualize:
                print(f'Predicted test acc {100. * local_correct/local_total}%')
                for img, pred, lab in zip(inputs_test, predicted_test, labels_test):
                    print("pred uniq: ", torch.unique(pred), "lab uniq: ", torch.unique(lab))
                    visualize_image_target_mask(
                        image=img.cpu(), target=pred.cpu(), labels=lab.cpu())

            # TODO - injected code
            # loss = dice_loss(outputs=outputs_test, labels=labels_test.float(), )/4
            loss = focal_loss(outputs=outputs_test, targets=labels_test, gamma=8)
            # 100 is to make the prob close to 1 after softmax
            labels_one_hot = 100*torch.nn.functional.one_hot(labels_test, num_classes=class_num).permute(0, 3,1,2).float()
            one_hot_loss = focal_loss(outputs = labels_one_hot, targets=labels_test, gamma=8)
            # torch.set_printoptions(profile="full")
            # for j, label_map in enumerate(labels_test):
            #     print(f"{j}: shape: {label_map.shape}")
            #     print(label_map)
            print(f'Rico: loss: {loss}, one hot loss: {one_hot_loss}')
            # i += 1
            # if i >= 1:
            #     break

    test_acc = 100. * correct_test / total_test
    print(f'Test Acc: {test_acc:.2f}%')

if __name__ == "__main__":
    # TODO
    # test_dataset = VOCSegmentationClass(image_set='train', year='2007')
    test_dataset = GTA5Dataset()
    # test_dataset = VOCSegmentationClass(image_set='trainval', year='2012')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers = 2,
        pin_memory = True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_num = len(test_dataset.classes)
    print("Loading state dict")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    print("Created barebone unet")
    model = UNet(class_num = class_num, 
                    intermediate_before_max_pool=INTERMEDIATE_BEFORE_MAX_POOL)
    print("Model loading")
    model.load_state_dict(state_dict)
    print("Eval model")
    model.to(device)
    eval_model(model=model, test_dataloader=test_dataloader, device=device, visualize=True, class_num=class_num)
