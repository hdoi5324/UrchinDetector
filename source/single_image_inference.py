import os


from glob import glob
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from datasets.utils_dataset import get_transform



def main() -> None:
    checkpoint_file = "../BenthicSynData/outputs/checkpoints/ckpt_od_urchin_v00_clahe_r50fpn_156_base03/checkpoint.pth"
    img_path = "../BenthicSynData/outputs/squidle_urchin_2009/PR*LC16.jpg"
    output_dir = "./outputs"
    device = "cuda"
    score_threshold = 0.5


    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    opt = DictConfig({'train':checkpoint['self.opt.train']})
    print(opt)

    from models.get_faster_rcnn_model import get_faster_rcnn_model as get_model
    model = get_model(opt)

    model.to(device)
    model.load_state_dict(checkpoint['model'])
    images = glob(img_path)
    for img_path in images:
        img_filename = os.path.basename(img_path)
        # Step 3: Apply inference preprocessing transforms
        img = Image.open(img_path).convert("RGB")
        transform = get_transform(False, opt.train)
        img, target = transform(img, None)
        # Load image code
        images = list(img.to(device) for img in [img])
        # Step 4: Use the models and visualize the prediction
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        prediction = outputs[0]

        labels = []
        selected_boxes = []
        boxes = prediction['boxes'].detach().cpu()
        for i in range(len(prediction['labels'])):
            score = prediction['scores'].detach().cpu()[i]
            name = str(int(prediction['labels'][i]))  # todo: add category name look up
            if score > score_threshold:
                labels.append(f"{name} {score:.2f}")
                selected_boxes.append(boxes[i, :].reshape((1, 4)))
        if len(selected_boxes) > 0:
            boxes = torch.concatenate(selected_boxes, dim=0)
        else:
            boxes = None
        del prediction, outputs, images


        img = read_image(img_path).detach()
        # Prediction boxes
        if boxes is not None:
            # Ground truth boxes
            img = draw_bounding_boxes(img, boxes=boxes,
                                      labels=labels,
                                      colors="orangered",
                                      width=7, font_size=16,
                                      font="/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf")

            img = to_pil_image(img)
            img.save(f"{output_dir}/{img_filename}")
            print(f"saving {img_filename}")

if __name__ == "__main__":
    main()
