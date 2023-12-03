from os import listdir
from os.path import join
import io
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from text_to_embedding import TextEmbeddingUsingBert

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        ToPILImage(),
        RandomCrop(512),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(512 // 4, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.input_images = sorted([join(dataset_dir + "/input_images", x) for x in listdir(dataset_dir + "/input_images") if is_image_file(x)])
        self.edited_images = sorted([join(dataset_dir + "/edited_images", x) for x in listdir(dataset_dir + "/edited_images") if is_image_file(x)])
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

        self.text_to_embeddings = TextEmbeddingUsingBert()

        self.text_to_embeddings = TextEmbeddingUsingBert()

        txt_file_path = dataset_dir + "/prompts.txt"
        with open(txt_file_path, 'r') as file:
            prompt_list = file.readlines()
        self.text_instructions = [ prompt.strip() for prompt in prompt_list ]

    def __getitem__(self, index):
        hr_image = self.hr_transform(ToTensor()(Image.open(self.edited_images[index])))
        lr_image = self.lr_transform(ToTensor()(Image.open(self.input_images[index])))
        text_prompt = self.text_to_embeddings.text_to_embedding(self.text_instructions[index])
        return lr_image, text_prompt, hr_image

    def __len__(self):
        return len(self.input_images)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.input_images = sorted([join(dataset_dir + "/input_images", x) for x in listdir(dataset_dir + "/input_images") if is_image_file(x)])
        self.edited_images = sorted([join(dataset_dir + "/edited_images", x) for x in listdir(dataset_dir + "/edited_images") if is_image_file(x)])
        self.text_to_embeddings = TextEmbeddingUsingBert()

        txt_file_path = dataset_dir + "/prompts.txt"
        with open(txt_file_path, 'r') as file:
            prompt_list = file.readlines()
        self.text_instructions = [ prompt.strip() for prompt in prompt_list ]

    def __getitem__(self, index):
        hr_image = ToTensor()(Image.open(self.edited_images[index]))
        channels, w, h = hr_image.size()
        lr_scale = Resize(512 // 4, interpolation=Image.BICUBIC)
        hr_scale = Resize(512, interpolation=Image.BICUBIC)
        real_image = ToTensor()(Image.open(self.input_images[index]))
        real_image = CenterCrop(512)(real_image)
        lr_image = lr_scale(real_image)
        hr_restore_img = hr_scale(lr_image)
        text_prompt = self.text_to_embeddings.text_to_embedding(self.text_instructions[index])
        return lr_image, hr_restore_img, hr_image, text_prompt

    def __len__(self):
        return len(self.input_images)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

        txt_file_path = dataset_dir + "/prompts.txt"
        with open(txt_file_path, 'r') as file:
            prompts_list = file.readlines()
        self.text_instructions = [ prompt.strip() for prompt in prompt_list ]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        text_prompt = self.text_instructions[index]
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image), text_prompt

    def __len__(self):
        return len(self.lr_filenames)
