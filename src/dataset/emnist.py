from torchvision import transforms, datasets

def get_dataset(split):
    dir = '../data/emnist'
    apply_transform_train = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    apply_transform_test = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.EMNIST(dir, split='letters', train=True, download=True,
                                        transform=apply_transform_train)
    test_dataset = datasets.EMNIST(dir, split='letters', train=False, download=True,
                                    transform=apply_transform_test)
    
    if split == 'train':
        return train_dataset
    else:
        return test_dataset